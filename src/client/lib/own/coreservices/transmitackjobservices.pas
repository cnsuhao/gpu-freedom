unit transmitackjobservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     jobqueuetables, servermanagers, loggers, identities, dbtablemanagers,
     workflowmanagers,
     SysUtils, Classes, DOM;


type TTransmitAckJobServiceThread = class(TTransmitServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
 protected
  procedure Execute; override;

 private
    jobqueuerow_   : TDbJobQueueRow;
    workflowman_   : TWorkflowManager;

    function  getPHPArguments() : AnsiString;
    procedure updateJobQueue();
end;

implementation

constructor TTransmitAckJobServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager;  var workflowman : TWorkflowManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitAckJobServiceThread]> ', conf, tableman);
 workflowman_ := workflowman;
end;

function TTransmitAckJobServiceThread.getPHPArguments() : AnsiString;
var rep : AnsiString;
begin
 rep :=     'nodeid='+encodeURL(myGPUId.nodeid)+'&';
 rep := rep+'nodename='+encodeURL(myGPUId.nodename)+'&';
 rep := rep+'jobqueueid='+encodeURL(jobqueuerow_.jobqueueid)+'&';
 rep := rep+'jobid='+encodeURL(jobqueuerow_.jobdefinitionid);

 logger_.log(LVL_DEBUG, logHeader_+'Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;

procedure TTransmitAckJobServiceThread.updateJobQueue();
begin
 jobqueuerow_.server_id := srv_.id;
 jobqueuerow_.ack_dt    := Now;

 //TODO: this has to be changed
 if workflowman_.getJobQueueWorkflow().changeStatusFromAcknowledgingToReady(jobqueuerow_) then
         logger_.log(LVL_DEBUG, logHeader_+'Jobqueue '+jobqueuerow_.jobqueueid+' set to READY.');
end;


procedure TTransmitAckJobServiceThread.Execute;
begin
 if not workflowman_.getJobQueueWorkflow().findRowInStatusWorkUnitRetrieved(jobqueuerow_) then
         begin
           logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status WORKUNIT_RETRIEVED. Exit.');
           done_      := True;
           erroneous_ := false;
           Exit;
         end;

 if not jobqueuerow_.requireack then
        begin
          logger_.log(LVL_WARNING, logHeader_+'Concurrency problem: found a job in status WORKUNIT_RETRIEVED which does not require acknowledgement.');
          done_      := True;
          erroneous_ := True;
          Exit;
        end
 else
         begin
           workflowman_.getJobQueueWorkflow().changeStatusFromWorkunitRetrievedToAcknowledging(jobqueuerow_);
           // Transmitting acknowledgement
           transmit('/jobqueue/ack_job.php?'+getPHPArguments(), false);
           if not erroneous_ then
                 begin
                   updateJobQueue();
                   finishTransmit('Job acknowledged :-)');
                 end
               else
                 begin
                   finishTransmit('Error in acknowledging job :-(');
                   workflowman_.getJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Problem in acknowledging job');
                 end;
         end;

 done_      := True;
end;


end.
