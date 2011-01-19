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
  constructor Create(var servMan : TServerManager; proxy, port : String;
                      var tableman : TDbTableManager;
                     var logger : TLogger; var conf : TCoreConfiguration;
                     var srv : TServerRecord);
 protected
    procedure Execute; override;

 private
   tableman_      : TDbTableManager;
   conf_          : TCoreConfiguration;
   srv_           : TServerRecord;

   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveJobServiceThread.Create(var servMan : TServerManager; proxy, port : String;
                                            var tableman : TDbTableManager;
                                            var logger : TLogger; var conf : TCoreConfiguration;
                                            var srv : TServerRecord);
begin
 inherited Create(servMan, proxy, port, logger);
 tableman_ := tableman_;
 conf_ := conf;
 srv_  := srv;
end;


procedure TReceiveJobServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbrow    : TDbJobRow;
    node     : TDOMNode;
    port     : String;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbrow.externalid  := StrToInt(node.FindNode('externalid').TextContent);
               dbrow.jobid       := node.FindNode('jobid').TextContent;
               dbrow.job         := node.FindNode('job').TextContent;
               dbrow.workunitincoming := node.FindNode('workunitincoming').TextContent;
               dbrow.workunitoutgoing := node.FindNode('workunitoutgoing').TextContent;
               dbrow.islocal     := false;
               dbrow.server_id   := srv_.id;
               dbrow.create_dt   := Now;
               dbrow.status      := JS_NEW;

               tableman_.getJobTable().insertOrUpdate(dbrow);
               logger_.log(LVL_DEBUG, 'Updated or added job with jobid: '+dbrow.jobid+' to TBJOB table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, '[TReceiveJobServiceThread]> Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;



procedure TReceiveJobServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive(srv_, '/cluster/jobqueue/get_jobs.php&nodeid='+encodeURL(myGPUId.NodeId),
         '[TReceiveJobServiceThread]> ', xmldoc, false);
 if not erroneous_ then
    begin
     parseXml(xmldoc);
     if not erroneous_ then
       begin
         //TODO: update tbjobqueue
       end;
    end;

 finishReceive(srv_, '[TReceiveJobServiceThread]> ', 'Service updated table TBJOB and TBJOBQUEUE succesfully :-)', xmldoc);
end;


end.
