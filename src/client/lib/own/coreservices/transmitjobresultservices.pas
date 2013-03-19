unit transmitjobresultservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     jobresulttables, jobqueuetables, servermanagers, loggers, identities, dbtablemanagers, workflowmanagers,
     SysUtils, Classes, DOM;


type TTransmitJobResultServiceThread = class(TTransmitServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager;
                     var jobresultrow : TDbJobResultRow);
 protected
  procedure Execute; override;

 private
    jobresultrow_ : TDbJobResultRow;
    jobqueuerow_  : TDbJobQueueRow;
    workflowman_  : TWorkflowManager;

    function  getPHPArguments() : AnsiString;
    procedure insertTransmission();
end;

implementation

constructor TTransmitJobResultServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager;  var jobresultrow : TDbJobResultRow);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitJobResultServiceThread]> ', conf, tableman);
 workflowman_  := workflowman;
 jobresultrow_ := jobresultrow;
end;

function TTransmitJobResultServiceThread.getPHPArguments() : AnsiString;
var rep : AnsiString;
begin
 rep := '';
 rep := rep+'jobqueueid='+encodeURL(jobresultrow_.jobqueueid)+'&';
 rep := rep+'jobid='+encodeURL(jobresultrow_.jobdefinitionid)+'&';
 rep := rep+'jobresultid='+encodeURL(jobresultrow_.jobresultid)+'&';
 rep := rep+'jobresult='+encodeURL(jobresultrow_.jobresult)+'&';
 rep := rep+'workunitresult='+encodeURL(jobresultrow_.workunitresult)+'&';
 rep := rep+'iserroneous='+encodeURL(BoolToStr(jobresultrow_.iserroneous))+'&';
 rep := rep+'errorid='+encodeURL(IntToStr(jobresultrow_.errorid))+'&';
 rep := rep+'errorarg='+encodeURL(jobresultrow_.errorarg)+'&';
 rep := rep+'errormsg='+encodeURL(jobresultrow_.errormsg)+'&';
 rep := rep+'nodeid='+encodeURL(myGPUId.nodeid)+'&';
 rep := rep+'nodename='+encodeURL(myGPUId.nodename);

 logger_.log(LVL_DEBUG, logHeader_+'Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;

procedure TTransmitJobResultServiceThread.insertTransmission();
begin
 jobresultrow_.server_id := srv_.id;
 tableman_.getJobResultTable().insertOrUpdate(jobresultrow_);
 logger_.log(LVL_DEBUG, logHeader_+'Updated or added '+IntToStr(jobresultrow_.id)+' to TBJOBRESULT table.');

  if workflowman_.getJobQueueWorkflow().changeStatusFromCompletedToTransmitted(jobqueuerow_) then
         logger_.log(LVL_DEBUG, logHeader_+'Jobqueue '+jobqueuerow_.jobqueueid+' set to TRANSMITTED.');
end;


procedure TTransmitJobResultServiceThread.Execute;
begin
  // retrieve jobqueue for this jobresult
  if not tableman_.getJobQueueTable().findRowWithJobQueueId(jobqueuerow_, jobresultrow_.jobqueueid) then
         begin
           logger_.log(LVL_SEVERE, logHeader_+'Could not find jobqueue with jobqueueid '+jobresultrow_.jobqueueid);
           Exit;
         end;

 transmit('/jobqueue/report_jobresult.php?'+getPHPArguments(), false);
 if not erroneous_ then
    begin
        insertTransmission();
    end;

 finishTransmit('Jobresult transmitted :-)');
end;

end.
