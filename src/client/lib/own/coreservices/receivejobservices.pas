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
    dbjobrow   : TDbJobDefinitionRow;
    dbqueuerow : TDbJobQueueRow;
    node       : TDOMNode;
    domjobdefinition     : TDOMNode;
    nodename,
    nodeid,
    jobdefinitionid      : String;
    requireack           : Boolean;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');

try
  begin
  node := xmldoc.DocumentElement.FirstChild;
  while Assigned(node) do
    begin
               jobdefinitionid := node.FindNode('jobdefinitionid').TextContent;
               dbqueuerow.jobdefinitionid := jobdefinitionid;
               dbjobrow.jobdefinitionid := jobdefinitionid;

               dbqueuerow.jobqueueid      := node.FindNode('jobqueueid').TextContent;;
               dbqueuerow.workunitjob     := node.FindNode('workunitjob').TextContent;
               dbqueuerow.workunitresult  := node.FindNode('workunitresult').TextContent;

               nodeid := node.FindNode('nodeid').TextContent;
               dbqueuerow.nodeid := nodeid;
               dbjobrow.nodeid   := nodeid;

               domjobdefinition := node.FindNode('jobdefinition');
               if domjobdefinition<>nil then
                  begin
                   dbjobrow.job := domjobdefinition.FindNode('job');
                   dbjobrow.jobtype := domjobdefinition.FindNode('jobtype');

                   nodename := domjobdefinition.FindNode('nodename').TextContent;
                   dbqueuerow.nodename := nodename;
                   dbjobrow.nodename := nodename;
                  end;

               requireack := (Trim(node.FindNode('requireack'))='1');
               dbjobrow.requireack := requireack;
               dbqueuerow.requireack:= requireack;

               //TODO: set this dates according to server
               dbqueuerow.create_dt   := Now;
               dbqueuerow.transmission_dt := Now;
               dbqueuerow.transmissionid  := node.FindNode('transmissionid');
               dbqueurow.ack_dt := nil;
               dbqueuerow.reception_dt := nil;

               dbqueuerow.server_id: = srv_.id;
               dbjobrow.server_id   := srv_.id;
               dbjobrow.status      := JS_READY;

               tableman_.getJobTable().insertOrUpdate(dbrow);
               //queuerow.job_id := dbrow.id;  // this could be setup at a later point
               tableman_.getJobQueueTable().insert(queuerow);
               logger_.log(LVL_DEBUG, logHeader_+'Updated or added job with jobdefinitionid: '+dbrow.jobdefinitionid+' to TBJOB and to TBJOBQUEUE table.');

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
 receive('/list_jobqueues.php?xml=1&crunch=1&nodeid='+encodeURL(myGPUId.NodeId),
         xmldoc, false);

 if not erroneous_ then
    begin
     parseXml(xmldoc);
    end;

 finishReceive('Service updated table TBJOB and TBJOBQUEUE succesfully :-)', xmldoc);
end;


end.
