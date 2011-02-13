unit transmitjobresultservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     jobresulttables, servermanagers, loggers, identities, dbtablemanagers,
     SysUtils, Classes, DOM;


type TTransmitJobResultServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var jobresultrow : TDbJobResultRow);
 protected
  procedure Execute; override;

 private
    jobresultrow_ : TDbJobResultRow;

    function  getPHPArguments() : AnsiString;
    procedure insertTransmission();
    procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TTransmitJobResultServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var jobresultrow : TDbJobResultRow);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitJobResultServiceThread]> ', conf, tableman);
 jobresultrow_ := jobresultrow;
end;

function TTransmitJobResultServiceThread.getPHPArguments() : AnsiString;
var rep : AnsiString;
begin
 rep :=     'nodeid='+encodeURL(myGPUId.nodeid)+'&';
 rep := rep+'nodename='+encodeURL(myGPUId.nodename)+'&';
 rep := rep+'requestid='+encodeURL(IntToStr(jobresultrow_.requestid))+'&';
 rep := rep+'jobid='+encodeURL(jobresultrow_.jobid)+'&';
 rep := rep+'jobresult='+encodeURL(jobresultrow_.jobresult)+'&';
 rep := rep+'workunitresult='+encodeURL(jobresultrow_.workunitresult)+'&';
 rep := rep+'iserroneous='+encodeURL(BoolToStr(jobresultrow_.iserroneous))+'&';
 rep := rep+'errorid='+encodeURL(IntToStr(jobresultrow_.errorid))+'&';
 rep := rep+'errormsg='+encodeURL(jobresultrow_.errormsg)+'&';
 rep := rep+'errorarg='+encodeURL(jobresultrow_.errorarg);

 logger_.log(LVL_DEBUG, logHeader_+'Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;

procedure TTransmitJobResultServiceThread.insertTransmission();
begin
 jobresultrow_.server_id := srv_.id;
 tableman_.getJobResultTable().insertOrUpdate(jobresultrow_);
 logger_.log(LVL_DEBUG, logHeader_+'Updated or added '+IntToStr(jobresultrow_.id)+' to TBJOBRESULT table.');
end;

procedure TTransmitJobResultServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    node     : TDOMNode;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               jobresultrow_.externalid  := StrToInt(node.FindNode('externalid').TextContent);
               logger_.log(LVL_DEBUG, 'Externalid for transmitted jobresult on server is: '+IntToStr(jobresultrow_.externalid));
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

procedure TTransmitJobResultServiceThread.Execute;
var xmldoc     : TXMLDocument;
    externalid : String;
begin
 receive('/jobqueue/report_jobresult_xml.php?'+getPHPArguments(), xmldoc, false);
 if not erroneous_ then
    begin
     parseXml(xmldoc);
     if not erroneous_ then
        insertTransmission();
    end;

 finishReceive('Jobresult transmitted :-)', xmldoc);
end;

end.
