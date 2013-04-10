unit transmitjobservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     jobdefinitiontables, jobqueuetables, servermanagers, loggers, identities,
     dbtablemanagers, workflowmanagers, jobapis,
     SysUtils, Classes, DOM;


type TTransmitJobServiceThread = class(TTransmitServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager); overload;
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager;
                     var trandetails : TJobTransmissionDetails); overload;

 protected
  procedure Execute; override;

 private
    workflowman_   : TWorkflowManager;
    jobqueuerow_   : TDbJobQueueRow;
    trandetails_   : TJobTransmissionDetails;

    function  getPHPArguments() : AnsiString;
end;

implementation

constructor TTransmitJobServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitJobServiceThread]> ', conf, tableman);
 workflowman_ := workflowman;
 trandetails_.nbrequests:=1;
 trandetails_.tagwujob:=false;
 trandetails_.tagwuresult:=false;
 trandetails_.workunitjob:='';
 trandetails_.workunitresult:='';
end;

constructor TTransmitJobServiceThread.CreateCreate(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                                                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager;
                                                   var trandetails : TJobTransmissionDetails);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitJobServiceThread]> ', conf, tableman);
 workflowman_ := workflowman;
 trandetails_ := trandetails;
end;

function TTransmitJobServiceThread.getPHPArguments() : AnsiString;
var rep : AnsiString;
begin
 rep :=     'nodeid='+encodeURL(myGPUId.nodeid)+'&';
 rep := rep+'nodename='+encodeURL(myGPUId.nodename)+'&';
 rep := rep+'jobid='+encodeURL(jobqueuerow_.jobdefinitionid)+'&';
 if trandetails_.nbrequests>1 then
   rep := rep+'jobqueueid='+encodeURL(jobqueuerow_.jobqueueid)+'&';
 rep := rep+'job='+encodeURL(jobqueuerow_.job)+'&';
 rep := rep+'workunitjob='+encodeURL(trandetails_.workunitjob)+'&';
 rep := rep+'workunitresult='+encodeURL(trandetails_.workunitresult)+'&';
 rep := rep+'nbrequests='+encodeURL(IntToStr(trandetails_.nbrequests))+'&';
 rep := rep+'tagwujob=';
 if trandetails_.tagwujob then rep := rep + '1&' else rep := rep + '0&';
 rep := rep+'tagwuresult=';
 if trandetails_.tagwuresult then rep := rep + '1&' else rep := rep + '0&';
 rep := rep+'requireack=';
 if jobqueuerow_.requireack then rep := rep + '1&' else rep := rep + '0&';
 rep := rep+'jobtype='+encodeURL(jobqueuerow_.jobtype);

 logger_.log(LVL_DEBUG, logHeader_+'Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;


procedure TTransmitJobServiceThread.Execute;
begin
 if not workflowman_.getServerJobQueueWorkflow().findRowInStatusForJobUpload(jobqueuerow_) then
         begin
           logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status S_FOR_JOB_UPLOAD. Exit.');
           done_      := True;
           erroneous_ := false;
           Exit;
         end;

 workflowman_.getServerJobQueueWorkflow().changeStatusFromForJobUploadToUploadingJob(jobqueuerow_);
 transmit('/jobqueue/report_job.php?'+getPHPArguments(), false);
 if not erroneous_ then
    begin
      workflowman_.getServerJobQueueWorkflow().changeStatusFromUploadingJobToForStatusRetrieval(jobqueuerow_);
      jobqueuerow_.serverstatus:='NEW';
      jobqueuerow_.update_dt:=Now;
      tableman_.getJobQueueTable().insertOrUpdate(jobqueuerow_);
      finishTransmit('Job transmitted :-)');
    end
 else
    begin
      workflowman_.getServerJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Could not transmit job to '+srv_.url);
      finishTransmit('Could not transmit job to '+srv_.url);
    end;

end;


end.
