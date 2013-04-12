unit receivejobresultservices;
{

  This unit receives a list of job results from GPU II superserver
   and stores it in the TDbJobResultTable object.

  (c) 2011-2013 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, dbtablemanagers,
     jobresulttables, jobqueuetables, loggers, downloadutils, coreconfigurations,
     workflowmanagers, Classes, SysUtils, DOM, identities;

type TReceiveJobResultServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);

 protected
    procedure Execute; override;

 private
   workflowman_  : TWorkflowManager;
   jobqueuerow_  : TDbJobQueueRow;

   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveJobResultServiceThread.Create(var servMan : TServerManager;proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
begin
 inherited Create(servMan, proxy, port, logger, '[TReceiveJobResultServiceThread]> ', conf, tableman);
 workflowman_ := workflowman;
end;

procedure TReceiveJobResultServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbrow              : TDbJobResultRow;
    node, wallnode     : TDOMNode;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;
try
  begin
   while Assigned(node) do
    begin
               dbrow.jobresultid     := node.FindNode('jobresultid').TextContent;
               dbrow.jobdefinitionid := node.FindNode('jobdefinitionid').TextContent;
               dbrow.jobqueueid      := node.FindNode('jobdefinitionid').TextContent;
               dbrow.jobresult      := node.FindNode('jobresult').TextContent;
               dbrow.workunitresult := node.FindNode('workunitresult').TextContent;
               dbrow.iserroneous    := (node.FindNode('iserroneous').TextContent='1');
               dbrow.errorid        := StrToInt(node.FindNode('errorid').TextContent);
               dbrow.errormsg       := node.FindNode('errormsg').TextContent;
               dbrow.errorarg       := node.FindNode('errorarg').TextContent;
               dbrow.nodeid         := node.FindNode('nodeid').TextContent;
               dbrow.nodename       := node.FindNode('nodename').TextContent;
               wallnode := node.FindNode('walltime');
               if Assigned(wallnode) then dbrow.walltime:=StrToIntDef(wallnode.TextContent, 0) else dbrow.walltime := 0;

               dbrow.server_id      := srv_.id;
               dbrow.walltime := 0;

               tableman_.getJobResultTable().insertOrUpdate(dbrow);

               logger_.log(LVL_DEBUG, 'Updated or added '+dbrow.jobresultid+' to TBJOBRESULT table.');

               node := node.NextSibling;
     end;  // while Assigned(node)
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



procedure TReceiveJobResultServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
  if not workflowman_.getServerJobQueueWorkflow().findRowInStatusForResultRetrieval(jobqueuerow_) then
         begin
           logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status S_FOR_RESULT_RETRIEVAL. Exit.');
           done_      := True;
           erroneous_ := false;
           Exit;
         end;

 workflowman_.getServerJobQueueWorkflow().changeStatusFromForResultRetrievalToRetrievingResult(jobqueuerow_);
 receive('/jobqueue/list_jobresults.php?xml=1&jobqueueid='+jobqueuerow_.jobqueueid, xmldoc, false);
 if not erroneous_ then
   begin
     parseXml(xmldoc);
     workflowman_.getServerJobQueueWorkflow().changeStatusFromRetrievingResultToCompleted(jobqueuerow_);
     finishReceive('Service updated table TBJOBRESULT table succesfully :-)', xmldoc);
   end
 else
   begin
     workflowman_.getServerJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Problem in retrieving result from server '+srv_.url);
     finishReceive('Problem in retrieving result from server '+srv_.url, xmldoc);
   end;

end;

end.
