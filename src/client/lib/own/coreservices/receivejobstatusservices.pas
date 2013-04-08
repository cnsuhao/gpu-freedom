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
    message     : String;
end;

type TReceiveJobstatusServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
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

constructor TReceiveJobstatusServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveJobstatusServiceThread]> ', conf, tableman);
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
  if Assigned(node) do
    begin
               status_.jobqueueid  := node.FindNode('jobqueueid').TextContent;
               status_.status      := node.FindNode('status').TextContent;
               status_.timestamp   := node.FindNode('timestamp').TextContent;
               status_.nodename    := node.FindNode('nodename').TextContent;
               status_.message     := node.FindNode('message').TextContent;

               jobqueuerow_.serverstatus:=status_.status;
               jobqueuerow_.update_dt:=Now;
               tableman_.getJobQueueTable().insertOrUpdate(jobqueuerow);

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
    found     : Boolean;
begin
 found := false;
 while not found do
    begin
      if not workflowman_.getServerJobQueueWorkflow().findRowInStatusForStatusRetrieval(jobqueuerow_) then
           begin
               logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status S_FOR_STATUS_RETRIEVAL. Exit.');
               done_      := True;
               erroneous_ := false;
               Exit;
           end;

    end;


 receive('/jobqueue/status_jobqueue.php?xml=1&jobqueueid='+encodeURL(),
         xmldoc, false);

 if not erroneous_ then
    begin
     parseXml(xmldoc);
     finishReceive('Service received status of TBJOBQUEUE on server successfully.', xmldoc);
    end
 else
    finishReceive('Communication problem in receiving status of TBJOBQUEUE on server.', xmldoc);

end;


end.
