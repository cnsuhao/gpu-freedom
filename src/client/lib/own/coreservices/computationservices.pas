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
      dbtablemanagers, jobqueuetables, jobresulttables, loggers, stkconstants,
      identities;

type
  TComputationServiceThread = class(TComputationThread)
   public
      
    constructor Create(var plugman : TPluginManager; var meth : TMethodController;
                       var res : TResultCollector; var frontman : TFrontendManager;
                       var workflowman : TWorkflowManager; var tableman : TDbTableManager;
                       var logger : TLogger);

    procedure   setSlot(var slot : Longint);

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
    slot_           : Longint; // slot in computationservicemanager

   end;

implementation

constructor TComputationServiceThread.Create(var plugman : TPluginManager; var meth : TMethodController;
                                             var res : TResultCollector; var frontman : TFrontendManager;
                                             var workflowman : TWorkflowManager; var tableman : TDbTableManager;
                                             var logger : TLogger);
begin
  workflowman_ := workflowman;
  tableman_    := tableman;
  logger_      := logger;
  logHeader_   := 'TComputationServiceThread> ';
  appPath_     := ExtractFilePath(ParamStr(0));
  slot_        := 1; // will be reset before the method call with setSlot

  inherited Create(plugman, meth, res, frontman, true); // waiting, will be activated by resum in compservicemanagers
end;

procedure TComputationServiceThread.setSlot(var slot : Longint);
begin
  slot_ := slot;
end;

procedure  TComputationServiceThread.Execute;
var parser : TJobParser;
    tempFolder : String;
begin
 erroneous_ := false;
 // job_ and thrdId_ need to be retrieved from TBJOBQUEUE
 if workflowman_.getClientJobQueueWorkflow().findRowInStatusReady(jobqueuerow_) then
    begin
       workflowman_.getClientJobQueueWorkflow().changeStatusFromReadyToRunning(jobqueuerow_);

       if (jobqueuerow_.workunitjobpath<>'') and (not FileExists(jobqueuerow_.workunitjobpath)) then
          begin
             workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Job workunit does not exist: '+jobqueuerow_.workunitjobpath);
             erroneous_ := true;
             done_ := true;
             Exit;
          end;


       tempFolder := appPath_+WORKUNIT_FOLDER+PathDelim+TEMP_WU_FOLDER;
       logger_.log(LVL_DEBUG, logHeader_+'Starting computation of job '+jobqueuerow_.job);
       logger_.log(LVL_DEBUG, logHeader_+'Incoming workunit is '+jobqueuerow_.workunitjobpath);
       logger_.log(LVL_DEBUG, logHeader_+'Outgoing workunit will be '+jobqueuerow_.workunitresultpath);
       logger_.log(LVL_DEBUG, logHeader_+'Temporary folder is '+tempFolder);

       try
          job_ := TJob.Create(jobqueuerow_.job, jobqueuerow_.workunitjobpath, jobqueuerow_.workunitresultpath);
          job_.stack.temporaryFolder:= tempFolder;

          parser := TJobParser.Create(plugman_, methController_, rescoll_, frontman_, job_, slot_);
          parser.parse();
          parser.Free;

       except
         on e : Exception do
             begin
               logger_.log(LVL_SEVERE, logHeader_+'Job threw exception '+e.Message);
               workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Job threw exception '+e.Message);
               erroneous_ := true;
               done_ := true;
               Exit;
             end;
       end;

       // now it is time to persist the result in TBJOBRESULT

       jobresultrow_.jobresultid := IntToStr(thrdid_);
       jobresultrow_.jobdefinitionid:=jobqueuerow_.jobdefinitionid;
       jobresultrow_.jobqueueid:=jobqueuerow_.jobqueueid;
       jobresultrow_.workunitresult:=jobqueuerow_.workunitresult;
       jobresultrow_.jobresult:=job_.JobResult;
       jobresultrow_.iserroneous:=job_.hasError;
       jobresultrow_.errorid:=job_.stack.error.ErrorID;
       jobresultrow_.errorarg:=job_.stack.error.ErrorArg;
       jobresultrow_.errormsg:=job_.stack.error.ErrorMsg;
       jobresultrow_.create_dt:=Now;
       jobresultrow_.server_id:= jobqueuerow_.server_id;
       jobresultrow_.nodeid   := myGPUID.nodeid;
       jobresultrow_.nodename := myGPUID.nodename;
       jobresultrow_.walltime := job_.computedTime;
       tableman_.getJobResultTable().insertOrUpdate(jobresultrow_);

       if job_.stack.workunitOutgoing<>jobqueuerow_.workunitresultpath then
          begin
             workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Job attempted to change outgoing workunit: '+jobqueuerow_.workunitresultpath+' * '+job_.stack.workunitOutgoing);
             erroneous_ := true;
             done_ := true;
             Exit;
          end;

       if (jobqueuerow_.workunitresultpath<>'') and (not FileExists(jobqueuerow_.workunitresultpath)) then
          begin
             workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Job did not create outgoing workunit: '+jobqueuerow_.workunitresultpath);
             erroneous_ := true;
             done_ := true;
             Exit;
          end;

       workflowman_.getClientJobQueueWorkflow().changeStatusFromRunningToComputed(jobqueuerow_);
       erroneous_ := job_.hasError;
       job_.Free;
    end    // findRowInStatusReady
 else logger_.log(LVL_DEBUG, logHeader_+'No jobqueue found in status READY');

 done_ := true;
end;


end.
