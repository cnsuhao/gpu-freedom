unit receivejobservices;
{

  This unit receives a list of jobs from GPU II server
   and stores it in the TDbJobTable object. An entry into TDbJobQueueTable
    is added as well.

  (c) 2011-2013 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, dbtablemanagers,
     jobdefinitiontables, jobqueuetables, jobqueuehistorytables, loggers, downloadutils, coreconfigurations,
     workflowmanagers, Classes, SysUtils, DOM, identities, synacode, stkconstants;

type TReceiveJobServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
protected
    procedure Execute; override;

 private
   appPath_     : String;
   workflowman_ : TWorkflowManager;

   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveJobServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var workflowman : TWorkflowManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveJobServiceThread]> ', conf, tableman);
 appPath_     := ExtractFilePath(ParamStr(0));
 workflowman_ := workflowman;
end;


procedure TReceiveJobServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbjobrow          : TDbJobDefinitionRow;
    dbqueuerow        : TDbJobQueueRow;
    dbqueuehistoryrow : TDbJobQueueHistoryRow;
    node,
    domjobdefinition  : TDOMNode;
    nodename,
    nodeid,
    jobdefinitionid,
    jobtype           : String;
    job               : AnsiString;
    requireack        : Boolean;

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

               dbqueuerow.islocal := false;
               dbjobrow.islocal := false;

               dbqueuerow.jobqueueid      := node.FindNode('jobqueueid').TextContent;;
               dbqueuerow.workunitjob     := Trim(node.FindNode('workunitjob').TextContent);
               dbqueuerow.workunitresult  := Trim(node.FindNode('workunitresult').TextContent);
               if dbqueuerow.workunitjob<>'' then
                 dbqueuerow.workunitjobpath:= appPath_+WORKUNIT_FOLDER+PathDelim+INCOMING_WU_FOLDER+PathDelim+dbqueuerow.workunitjob
               else  dbqueuerow.workunitjobpath:='';
               if dbqueuerow.workunitresult<>'' then
                 dbqueuerow.workunitresultpath:=appPath_+WORKUNIT_FOLDER+PathDelim+OUTGOING_WU_FOLDER+PathDelim+dbqueuerow.workunitjob
               else
                 dbqueuerow.workunitresultpath:='';

               nodeid := node.FindNode('nodeid').TextContent;
               dbqueuerow.nodeid := nodeid;
               dbjobrow.nodeid   := nodeid;

               domjobdefinition := node.FindNode('jobdefinition');
               if domjobdefinition<>nil then
                  begin
                   job := domjobdefinition.FindNode('job').TextContent;
                   jobtype := domjobdefinition.FindNode('jobtype').TextContent;
                   dbjobrow.job     := job;
                   dbjobrow.jobtype := jobtype;
                   dbqueuerow.job     := job;
                   dbqueuerow.jobtype := jobtype;

                   nodename := domjobdefinition.FindNode('nodename').TextContent;
                   dbqueuerow.nodename := nodename;
                   dbjobrow.nodename := nodename;
                  end;

               requireack := (Trim(node.FindNode('requireack').TextContent)='1');
               dbjobrow.requireack := requireack;
               dbqueuerow.requireack:= requireack;

               dbqueuerow.create_dt   := Now;
               dbqueuerow.update_dt   := dbqueuerow.create_dt;
               dbqueuerow.transmission_dt := Now;
               dbqueuerow.transmissionid  := node.FindNode('transmissionid').TextContent;
               dbqueuerow.ack_dt := 0;
               dbqueuerow.acknodeid := '';
               dbqueuerow.acknodename := '';
               dbqueuerow.reception_dt := 0;

               dbqueuerow.server_id := srv_.id;
               dbjobrow.server_id   := srv_.id;
               dbqueuerow.status    := C_NEW;

               // jobqueuehistoryrow
               dbqueuehistoryrow.status     := dbqueuerow.status;
               dbqueuehistoryrow.jobqueueid := dbqueuerow.jobqueueid;
               dbqueuehistoryrow.message    := logHeader_+'Received a new job from server '+srv_.url;

               tableman_.getJobDefinitionTable().insertOrUpdate(dbjobrow);
               tableman_.getJobQueueTable().insertOrUpdate(dbqueuerow);
               tableman_.getJobQueueHistoryTable().insert(dbqueuehistoryrow);
               logger_.log(LVL_DEBUG, logHeader_+'Updated or added job with jobdefinitionid: '+dbjobrow.jobdefinitionid+' to TBJOBDEFINITION and to TBJOBQUEUE table.');

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
 receive('/jobqueue/list_jobqueues.php?xml=1&crunch=1&nodeid='+encodeURL(myGPUId.NodeId),
         xmldoc, false);

 if not erroneous_ then
    begin
     parseXml(xmldoc);
    end;

 finishReceive('Service updated table TBJOB and TBJOBQUEUE succesfully :-)', xmldoc);
end;


end.
