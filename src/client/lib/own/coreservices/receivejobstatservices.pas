unit receivejobstatservices;
{

  This unit receives statistics about jobs from GPU II server
   and stores it in the TDbJobStatsTable object.

  (c) 2011-2013 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, jobstatstables, dbtablemanagers,
     jobdefinitiontables, loggers, downloadutils, coreconfigurations,
     Classes, SysUtils, DOM, identities, synacode;

type TReceiveJobstatServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);
protected
    procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveJobstatServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveJobstatServiceThread]> ', conf, tableman);
end;


procedure TReceiveJobstatServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbstatrow   : TDbJobStatsRow;
    node        : TDOMNode;
    domjobdefinition     : TDOMNode;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');

try
  begin
  node := xmldoc.DocumentElement.FirstChild;
  while Assigned(node) do
    begin
               dbstatrow.jobdefinitionid := node.FindNode('jobdefinitionid').TextContent;
               dbstatrow.job             := node.FindNode('job').TextContent;
               dbstatrow.jobtype         := node.FindNode('jobtype').TextContent;
               dbstatrow.requireack      := (node.FindNode('requireack').TextContent = '1');
               dbstatrow.requests        := StrToIntDef(node.FindNode('requests').TextContent, 0);
               dbstatrow.transmitted     := StrToIntDef(node.FindNode('transmitted').TextContent, 0);
               dbstatrow.received        := StrToIntDef(node.FindNode('received').TextContent, 0);
               dbstatrow.acknowledged    := StrToIntDef(node.FindNode('acknowledged').TextContent, 0);
               dbstatrow.server_id       := srv_.id;
               dbstatrow.create_dt       := Now;

               tableman_.getJobstatsTable().insertOrUpdate(dbstatrow);
               //queuerow.job_id := dbrow.id;  // this could be setup at a later point
               logger_.log(LVL_DEBUG, logHeader_+'Updated or added jobstat with jobdefinitionid: '+dbstatrow.jobdefinitionid+' to TBJOBSTATS table.');

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



procedure TReceiveJobStatServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/list_jobstats.php?xml=1&nodeid='+encodeURL(myGPUId.NodeId),
         xmldoc, false);

 if not erroneous_ then
    begin
     parseXml(xmldoc);
    end;

 finishReceive('Service updated table TBJOBSTATS succesfully :-)', xmldoc);
end;


end.
