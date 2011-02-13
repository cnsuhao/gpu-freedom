unit receivejobresultservices;
{

  This unit receives a list of job results from GPU II superserver
   and stores it in the TDbJobResultTable object.

  (c) 2011 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, dbtablemanagers,
     jobresulttables, loggers, downloadutils, coreconfigurations,
     Classes, SysUtils, DOM, identities;

type TReceiveJobResultServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; jobid : String);

 protected
    procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);

   jobid_  : String;
   job_id_ : Longint;
end;

implementation

constructor TReceiveJobResultServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; jobid : String);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveJobResultServiceThread]> ', conf, tableman);
 jobid_  := jobid;
 job_id_ := tableman_.getJobTable().getId(jobid_);
end;

procedure TReceiveJobResultServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbrow    : TDbJobResultRow;
    node     : TDOMNode;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbrow.externalid     := StrToInt(node.FindNode('externalid').TextContent);
               dbrow.job_id         := job_id_;
               dbrow.jobid          := jobid_;
               dbrow.jobresult      := node.FindNode('jobresult').TextContent;
               dbrow.workunitresult := node.FindNode('workunitresult').TextContent;
               dbrow.iserroneous    := (node.FindNode('iserroneous').TextContent='1');
               dbrow.errorid        := StrToInt(node.FindNode('errorid').TextContent);
               dbrow.errormsg       := node.FindNode('errormsg').TextContent;
               dbrow.errorarg       := node.FindNode('errorarg').TextContent;
               dbrow.nodeid         := node.FindNode('nodeid').TextContent;
               dbrow.nodename       := node.FindNode('nodename').TextContent;
               dbrow.server_id      := srv_.id;
               tableman_.getJobResultTable().insertOrUpdate(dbrow);
               logger_.log(LVL_DEBUG, 'Updated or added '+IntToStr(dbrow.externalid)+' to TBJOBRESULT table.');
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



procedure TReceiveJobResultServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/jobqueue/get_jobresults_xml.php?jobid='+jobid_, xmldoc, false);
 if not erroneous_ then
     parseXml(xmldoc);

 finishReceive('Service updated table TBJOBRESULT table succesfully :-)', xmldoc);
end;

end.
