unit receivejobservices;
{

  This unit receives a list of jobs from GPU II server
   and stores it in the TDbJobTable object. An entry into TDbJobQueueTable
    is added as well.

  (c) 2011-2013 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, jobqueuetables, dbtablemanagers,
     jobdefinitiontables, loggers, downloadutils, coreconfigurations,
     Classes, SysUtils, DOM, identities, synacode;

type TReceiveJobServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);
protected
    procedure Execute; override;


 private
   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveJobServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveJobServiceThread]> ', conf, tableman);
end;


procedure TReceiveJobServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbjobrow : TDbJobDefinitionRow;
    queuerow : TDbJobQueueRow;
    node     : TDOMNode;
    port     : String;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

try
  begin
   while Assigned(node) do
    begin
               dbjobrow.externalid   := node.FindNode('externalid').TextContent;
               dbjobrow.jobid        := node.FindNode('jobid').TextContent;
               queuerow.requestid := StrToInt(node.FindNode('requestid').TextContent);
               queuerow.server_id := srv_.id;
               dbjobrow.job              := node.FindNode('job').TextContent;
               dbjobrow.workunitincoming := node.FindNode('workunitincoming').TextContent;
               dbjobrow.workunitoutgoing := node.FindNode('workunitoutgoing').TextContent;
               dbjobrow.requests    := StrToInt(node.FindNode('requests').TextContent);
               dbjobrow.delivered   := StrToInt(node.FindNode('delivered').TextContent);
               dbjobrow.results     := StrToInt(node.FindNode('results').TextContent);
               dbjobrow.nodeid      := node.FindNode('nodeid').TextContent;
               dbjobrow.nodename    := node.FindNode('nodename').TextContent;
               dbjobrow.islocal     := false;
               dbjobrow.server_id   := srv_.id;
               dbjobrow.create_dt   := Now;
               dbjobrow.status      := JS_READY;

               tableman_.getJobTable().insertOrUpdate(dbrow);
               queuerow.job_id := dbrow.id;
               tableman_.getJobQueueTable().insert(queuerow);
               logger_.log(LVL_DEBUG, logHeader_+'Updated or added job with externalid: '+dbrow.externalid+' to TBJOB and to TBJOBQUEUE table.');

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



procedure TReceiveJobServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/jobqueue/get_jobs_xml.php?max=3&nodeid='+encodeURL(myGPUId.NodeId),
         xmldoc, false);

 if not erroneous_ then
    begin
     parseXml(xmldoc);
    end;

 finishReceive('Service updated table TBJOB and TBJOBQUEUE succesfully :-)', xmldoc);
end;


end.
