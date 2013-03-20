unit computationservices;
{
    TComputationService is a database aware ComputationThread which executes a job stored
    in TBJOBQUEUE. It first tries to retrieve a jobqueue entry in status READY,
    sets jobqueue to RUNNING. When the job is computed, it persists an entry in TBJOBRESULT
    and sets the jobqueue to COMPUTED.

  (c) by 2002-2013 HB9TVM and the GPU Development Team
   This unit is released under GNU Public License (GPL)
}
interface

uses  Classes, Sysutils,
      jobs, methodcontrollers, pluginmanagers, resultcollectors, frontendmanagers,
      jobparsers, managedthreads, computationthreads, workflowmanagers,
      dbtablemanagers, jobqueuetables, jobresulttables, loggers, stkconstants;

type
  TComputationServiceThread = class(TComputationThread)
   public
      
    constructor Create(var plugman : TPluginManager; var meth : TMethodController;
                       var res : TResultCollector; var frontman : TFrontendManager;
                       var workflowman : TWorkflowManager; var tableman : TDbTableManager;
                       var logger : TLogger);

   protected
    procedure Execute; override;

   protected
    workflowman_    : TWorkflowManager;
    tableman_       : TDbTableManager;
    jobqueuerow_    : TDbJobQueueRow;
    jobresultrow_   : TDbJobResultRow;
    logger_         : TLogger;
    logHeader_      : String;
    appPath_        : String;

   end;

implementation

constructor TComputationServiceThread.Create(var plugman : TPluginManager; var meth : TMethodController;
                                             var res : TResultCollector; var frontman : TFrontendManager;
                                             var workflowman : TWorkflowManager; var tableman : TDbTableManager;
                                             var logger : TLogger);
begin
  inherited Create(plugman, meth, res, frontman); // running
  workflowman_ := workflowman;
  tableman_    := tableman;
  logger_      := logger;
  logHeader_   := 'TComputationServiceThread> ';
  appPath_     := ExtractFilePath(ParamStr(0));
end;


procedure  TComputationServiceThread.Execute;
var parser : TJobParser;
    tempFolder : String;
begin
 erroneous_ := false;
 // job_ and thrdId_ need to be retrieved from TBJOBQUEUE
 if workflowman_.getJobQueueWorkflow().findRowInStatusReady(jobqueuerow_) then
    begin
       workflowman_.getJobQueueWorkflow().changeStatusFromReadyToRunning(jobqueuerow_);

       thrdid_ := Round(Random(1000000)); // TODO: check what is this used for

       tempFolder := appPath_+PathDelim+WORKUNIT_FOLDER+PathDelim+TEMP_WU_FOLDER;
       job_ := TJob.Create(jobqueuerow_.job, jobqueuerow_.workunitjob, jobqueuerow_.workunitresult);
       job_.stack.temporaryFolder:= tempFolder;

       logger_.log(LVL_DEBUG, logHeader_+'Starting computation of job '+jobqueuerow_.job);
       logger_.log(LVL_DEBUG, logHeader_+'Incoming workunit is '+jobqueuerow_.workunitjob);
       logger_.log(LVL_DEBUG, logHeader_+'Outgoing workunit will be '+jobqueuerow_.workunitjob);
       logger_.log(LVL_DEBUG, logHeader_+'Temporary folder is '+tempFolder);

       parser := TJobParser.Create(plugman_, methController_, rescoll_, frontman_, job_, thrdId_);
       parser.parse();
       parser.Free;

       erroneous_ := job_.hasError;

       // now it is time to

       workflowman_.getJobQueueWorkflow().changeStatusFromRunningToCompleted(jobqueuerow_);

       job_.Free;
    end
 else logger_.log(LVL_DEBUG, logHeader_+'No jobqueue found in status READY');

 done_ := true;
end;


end.
