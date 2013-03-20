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

uses  Classes,
      jobs, methodcontrollers, pluginmanagers, resultcollectors, frontendmanagers,
      jobparsers, managedthreads, computationthreads, workflowmanagers,
      dbtablemanagers, jobqueuetables, jobresulttables, loggers;

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
end;


procedure  TComputationServiceThread.Execute;
var parser : TJobParser;
begin
 erroneous_ := false;
 // job_ and thrdId_ need to be retrieved from TBJOBQUEUE
 if workflowman_.getJobQueueWorkflow().findRowInStatusReady(jobqueuerow_) then
    begin
       thrdid_ := Round(Random(1000000)); // TODO: check what is this used for
       job_ := TJob.Create(jobqueuerow_.job, jobqueuerow_.workunitjob, jobqueuerow_.workunitresult);
       logger_.log(LVL_DEBUG, logHeader_+'Starting computation of job '+jobqueuerow_.job);
       logger_.log(LVL_DEBUG, logHeader_+'Incoming workunit is '+jobqueuerow_.workunitjob);
       logger_.log(LVL_DEBUG, logHeader_+'Outgoing workunit will be '+jobqueuerow_.workunitjob);

       parser := TJobParser.Create(plugman_, methController_, rescoll_, frontman_, job_, thrdId_);
       parser.parse();
       parser.Free;

       erroneous_ := job_.hasError;
       job_.Free;
    end
 else logger_.log(LVL_DEBUG, logHeader_+'No jobqueue found in status READY');

 done_ := true;
end;


end.
