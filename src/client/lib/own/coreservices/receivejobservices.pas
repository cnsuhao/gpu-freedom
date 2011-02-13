unit receivejobservices;
{

  This unit receives a list of jobs from GPU II server
   and stores it in the TDbJobTable object. An entry into TDbJobQueueTable
    is added as well.

  (c) 2011 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, jobqueuetables, dbtablemanagers,
     jobtables, loggers, downloadutils, coreconfigurations,
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
    dbrow    : TDbJobRow;
    queuerow : TDbJobQueueRow;
    node     : TDOMNode;
    port     : String;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbrow.externalid   := node.FindNode('externalid').TextContent;
               dbrow.jobid        := node.FindNode('jobid').TextContent;
               queuerow.requestid := StrToInt(node.FindNode('requestid').TextContent);
               queuerow.server_id := srv_.id;
               dbrow.job              := node.FindNode('job').TextContent;
               dbrow.workunitincoming := node.FindNode('workunitincoming').TextContent;
               dbrow.workunitoutgoing := node.FindNode('workunitoutgoing').TextContent;
               dbrow.requests    := StrToInt(node.FindNode('requests').TextContent);
               dbrow.delivered   := StrToInt(node.FindNode('delivered').TextContent);
               dbrow.results     := StrToInt(node.FindNode('results').TextContent);
               dbrow.nodeid      := node.FindNode('nodeid').TextContent;
               dbrow.nodename    := node.FindNode('nodename').TextContent;
               dbrow.islocal     := false;
               dbrow.server_id   := srv_.id;
               dbrow.create_dt   := Now;
               dbrow.status      := JS_RECEIVED;

               tableman_.getJobTable().insertOrUpdate(dbrow);
               queuerow.job_id := dbrow.id;
               tableman_.getJobQueueTable().insert(queuerow);
               logger_.log(LVL_DEBUG, logHeader_+'Updated or added job with externalid: '+dbrow.externalid+' to TBJOB and to TBJOBQUEUE table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, logHeader_+'Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)

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
