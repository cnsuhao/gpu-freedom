unit receivejobstatusservices;
{

  This unit receives the status about a single job submitted to a GPU II server.
  The service takes one jobqueue in status S_FOR_STATUS_RETRIEVAL and retrieves
   its status on the server.

  Depending on the status, a transition is executed:
  if status on server is COMPLETED: transition either to S_FOR_WU_RETRIEVAL (if a workunit needs to be retrieved)
                                    or to S_FOR_RESULT_RETRIEVAL
  if status on server is ERROR: transition to S_ERROR logging error message
  if status on server in (NEW, TRANSMITTED, ACKNOWLEDGED): transition to S_FOR_STATUS_RETRIEVAL


  (c) 2011-2013 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, jobstatstables, dbtablemanagers,
     jobdefinitiontables, loggers, downloadutils, coreconfigurations,
     workflowmanagers, jobqueuetables,
     Classes, SysUtils, DOM, identities, synacode;

// matches structure on server produced by jobqueue/status_jobqueue.php
type TJobStatus = record
    jobqueueid,
    status,
    timestamp,
    nodename,
    message,
    jobresultid  : String;
end;

type TReceiveJobstatusServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
protected
    workflowman_ : TWorkflowManager;
    jobqueuerow_ : TDbJobQueueRow;
    status_      : TJobStatus;

    procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveJobstatusServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
begin
 inherited Create(servMan, proxy, port, logger, '[TReceiveJobstatusServiceThread]> ', conf, tableman);
 workflowman_ := workflowman;
end;


procedure TReceiveJobstatusServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    node        : TDOMNode;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');

try
  begin
  node := xmldoc.DocumentElement.FirstChild;
  if Assigned(node) then
    begin
               status_.jobqueueid  := node.FindNode('jobqueueid').TextContent;
               status_.status      := node.FindNode('status').TextContent;
               status_.timestamp   := node.FindNode('timestamp').TextContent;
               status_.nodename    := node.FindNode('nodename').TextContent;
               status_.message     := node.FindNode('message').TextContent;
               status_.jobresultid := node.FindNode('jobresultid').TextContent;

               workflowman_.getServerJobQueueWorkflow().changeStatusFromRetrievingStatusToStatusRetrieved(jobqueuerow_, 'status='+status_.status+', nodename='+status_.nodename+', timestamp='+status_.timestamp+
                                                                                                                        ', message='+status_.message+', jobresultid='+status_.jobresultid);

               if status_.status='ERROR' then
                  workflowman_.getServerJobQueueWorkflow.changeStatusToError(jobqueuerow_, status_.message)
               else
               if status_.status='COMPLETED' then
                  begin
                     jobqueuerow_.jobresultid:=status_.jobresultid;

                     if Trim(jobqueuerow_.workunitresultpath)='' then
                        workflowman_.getServerJobQueueWorkflow().changeStatusFromStatusRetrievedToForResultRetrieval(jobqueuerow_, 'Fast transition, no workunit to be retrieved')
                     else
                        workflowman_.getServerJobQueueWorkflow().changeStatusFromStatusRetrievedToForWuRetrieval(jobqueuerow_);
                  end;

               jobqueuerow_.serverstatus:=status_.status;
               jobqueuerow_.update_dt:=Now;
               tableman_.getJobQueueTable().insertOrUpdate(jobqueuerow_);

     end;  // if Assigned(node)
end; // try
except
    on E : Exception do
       begin
            erroneous_ := true;
            logger_.log(LVL_SEVERE, logHeader_+'Exception catched in parseXML: '+E.Message);
       end;
end; // except

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;



procedure TReceiveJobStatusServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
  if not workflowman_.getServerJobQueueWorkflow().findRowInStatusForStatusRetrieval(jobqueuerow_) then
           begin
               logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status S_FOR_STATUS_RETRIEVAL. ');

               while workflowman_.getServerJobQueueWorkflow().findRowInStatusStatusRetrieved(jobqueuerow_) do
                     begin
                        workflowman_.getServerJobQueueWorkflow().changeStatusFromStatusRetrievedToForStatusRetrieval(jobqueuerow_, 'queuing request again, as job is not completed yet');
                     end;
               logger_.log(LVL_DEBUG, logHeader_+'All jobs (if any) in S_FOR_STATUS_RETRIEVED reset to S_FOR_STATUS_RETRIEVAL.');

               done_      := True;
               erroneous_ := false;
               Exit;
           end;


 setServer(jobqueuerow_.server_id);
 receive('/jobqueue/status_jobqueue.php?xml=1&jobqueueid='+encodeURL(jobqueuerow_.jobqueueid), xmldoc, false);

 if not erroneous_ then
    begin
     parseXml(xmldoc);
     finishReceive('Service received status of TBJOBQUEUE on server successfully.', xmldoc);
    end
 else
    begin
     finishReceive('Communication problem in receiving status of TBJOBQUEUE on server.', xmldoc);
     workflowman_.getServerJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Communication problem in receiving status of TBJOBQUEUE on server.');
    end;
end;


end.
