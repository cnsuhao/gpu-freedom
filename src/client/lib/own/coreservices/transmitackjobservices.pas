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
 jobqueuerow_.serverstatus:='ACKNOWLEDGED';
 jobqueuerow_.update_dt:=Now;

 if workflowman_.getClientJobQueueWorkflow().changeStatusFromAcknowledgingToReady(jobqueuerow_) then
         logger_.log(LVL_DEBUG, logHeader_+'Jobqueue '+jobqueuerow_.jobqueueid+' set to READY.');
end;


procedure TTransmitAckJobServiceThread.Execute;
begin
 if not workflowman_.getClientJobQueueWorkflow().findRowInStatusForAcknowledgement(jobqueuerow_) then
         begin
           logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status FOR_ACKNOWLEDGEMENT. Exit.');
           done_      := True;
           erroneous_ := false;
           Exit;
         end;

 if not jobqueuerow_.requireack then
        begin
          logger_.log(LVL_WARNING, logHeader_+'Found a job in status FOR_ACKNOWLEDGEMENT which does not require acknowledgement.');
          workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Found a job in status FOR_ACKNOWLEDGEMENT which does not require acknowledgement.');
          done_      := True;
          erroneous_ := True;
          Exit;
        end
 else
         begin
           workflowman_.getClientJobQueueWorkflow().changeStatusFromForAcknowledgementToAcknowledging(jobqueuerow_);
           // Transmitting acknowledgement
           transmit('/jobqueue/ack_job.php?'+getPHPArguments(), false);
           if not erroneous_ then
                 begin
                   updateJobQueue();
                   finishTransmit('Job acknowledged :-)');
                 end
               else
                 begin
                   workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Communication problem in acknowledging job');
                   finishTransmit('Error in acknowledging job :-(');
                 end;
         end;

 done_      := True;
end;


end.
