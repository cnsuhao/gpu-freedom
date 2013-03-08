unit transmitjobservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     jobdefinitiontables, servermanagers, loggers, identities, dbtablemanagers,
     SysUtils, Classes, DOM;


type TTransmitJobServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager; var jobdefinitionrow : TDbJobDefinitionRow;
                     var trandetails : TJobTransmissionDetails);
 protected
  procedure Execute; override;

 private
    jobdefrow_   : TDbJobDefinitionRow;
    trandetails_ : TJobTransmissionDetails;

    function  getPHPArguments() : AnsiString;
    procedure insertTransmission();
end;

implementation

constructor TTransmitJobServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager; var jobdefinitionrow : TDbJobDefinitionRow;
                   var trandetails : TJobTransmissionDetails);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitJobServiceThread]> ', conf, tableman);
 jobdefrow_ := jobdefinitionrow;
 trandetails_ := trandetails;
end;

function TTransmitJobServiceThread.getPHPArguments() : AnsiString;
var rep : AnsiString;
begin
 rep :=     'xml=1&nodeid='+encodeURL(myGPUId.nodeid)+'&';
 rep := rep+'nodename='+encodeURL(myGPUId.nodename)+'&';
 rep := rep+'jobid='+encodeURL(jobdefrow_.jobdefinitionid)+'&';
 rep := rep+'job='+encodeURL(jobdefrow_.job)+'&';
 rep := rep+'workunitjob='+encodeURL(trandetails_.workunitjob)+'&';
 rep := rep+'workunitresult='+encodeURL(trandetails_.workunitresult)+'&';
 rep := rep+'nbrequests='+encodeURL(IntToStr(trandetails_.nbrequests))+'&';
 rep := rep+'tagwujob=';
 if trandetails_.tagwujob then rep := rep + '1&' else rep := rep + '0&';
 rep := rep+'tagwuresult=';
 if trandetails_.tagwuresult then rep := rep + '1&' else rep := rep + '0&';
 rep := rep+'requireack=';
 if jobdefrow_.requireack then rep := rep + '1&' else rep := rep + '0&';
 rep := rep+'jobtype='+encodeURL(jobdefrow_.jobtype);

 logger_.log(LVL_DEBUG, logHeader_+'Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;

procedure TTransmitJobServiceThread.insertTransmission();
begin
 jobdefrow_.server_id := srv_.id;
 tableman_.getJobDefinitionTable().insertOrUpdate(jobdefrow_);
 logger_.log(LVL_DEBUG, logHeader_+'Updated or added job with jobid '+jobdefrow_.jobdefinitionid+' to TBJOBDEFINITION table.');
end;


procedure TTransmitJobServiceThread.Execute;
var xmldoc     : TXMLDocument;
    externalid : String;
begin
 receive('/jobqueue/report_job.php?'+getPHPArguments(), xmldoc, false);
 if not erroneous_ then
    begin
        insertTransmission();
    end;

 finishReceive('Job transmitted :-)', xmldoc);
end;


end.
