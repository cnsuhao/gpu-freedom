unit transmitjobservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     jobtables, servermanagers, loggers, identities, dbtablemanagers,
     SysUtils, Classes, DOM;


type TTransmitJobServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var jobrow : TDbJobRow);
 protected
  procedure Execute; override;

 private
    jobrow_ : TDbJobRow;

    function  getPHPArguments() : AnsiString;
    procedure insertTransmission();
    procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TTransmitJobServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var jobrow : TDbJobRow);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitJobServiceThread]> ', conf, tableman);
 jobrow_ := jobrow;
end;

function TTransmitJobServiceThread.getPHPArguments() : AnsiString;
var rep : AnsiString;
begin
 rep :=     'nodeid='+encodeURL(myGPUId.nodeid)+'&';
 rep := rep+'nodename='+encodeURL(myGPUId.nodename)+'&';
 rep := rep+'jobid='+encodeURL(jobrow_.jobid)+'&';
 rep := rep+'job='+encodeURL(jobrow_.job)+'&';
 rep := rep+'workunitincoming='+encodeURL(jobrow_.workunitincoming)+'&';
 rep := rep+'workunitoutgoing='+encodeURL(jobrow_.workunitoutgoing)+'&';
 rep := rep+'requests='+encodeURL(IntToStr(jobrow_.requests));

 logger_.log(LVL_DEBUG, logHeader_+'Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;

procedure TTransmitJobServiceThread.insertTransmission();
begin
 jobrow_.server_id := srv_.id;
 jobrow_.status    := JS_TRANSMITTED;
 tableman_.getJobTable().insertOrUpdate(jobrow_);
 logger_.log(LVL_DEBUG, logHeader_+'Updated or added '+IntToStr(jobrow_.id)+' to TBJOB table.');
end;

procedure TTransmitJobServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    node     : TDOMNode;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               jobrow_.externalid  := node.FindNode('externalid').TextContent;
               logger_.log(LVL_DEBUG, 'Externalid for transmitted job on server is: '+jobrow_.externalid);
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

procedure TTransmitJobServiceThread.Execute;
var xmldoc     : TXMLDocument;
    externalid : String;
begin
 receive('/jobqueue/report_job_xml.php?'+getPHPArguments(), xmldoc, false);
 if not erroneous_ then
    begin
     parseXml(xmldoc);
     if not erroneous_ then
        insertTransmission();
    end;

 finishReceive('Job transmitted :-)', xmldoc);
end;


end.
