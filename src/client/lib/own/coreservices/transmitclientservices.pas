unit transmitclientservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     servermanagers, loggers, identities, SysUtils, Classes;


type TTransmitClientServiceThread = class(TTransmitServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration);
 protected
  procedure Execute; override;

 private
    conf_ : TCoreConfiguration;
    function getReportString : AnsiString;
end;



implementation

constructor TTransmitClientServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                                                var conf : TCoreConfiguration);
begin
 inherited Create(servMan, proxy, port, logger);
 conf_ := conf;
end;

function TTransmitClientServiceThread.getReportString : AnsiString;
var rep : AnsiString;
begin
with myGPUID do
 begin
  rep :=     'nodename='+encodeURL(nodename)+'&';
  rep := rep+'nodeid='+encodeURL(nodeid)+'&';
  rep := rep+'country='+encodeURL(country)+'&';
  rep := rep+'region='+encodeURL(region)+'&';
  rep := rep+'city='+encodeURL(city)+'&';
  rep := rep+'zip='+encodeURL(zip)+'&';
  rep := rep+'uptime='+encodeURL(FloatToStr(uptime))+'&';
  rep := rep+'totaluptime='+encodeURL(FloatToStr(totaluptime))+'&';
  rep := rep+'ip='+encodeURL(ip)+'&';
  rep := rep+'localip='+encodeURL(localip)+'&';
  rep := rep+'port='+encodeURL(port)+'&';
  rep := rep+'acceptincoming='+encodeURL('0')+'&';
  rep := rep+'cputype='+encodeURL(cputype)+'&';
  rep := rep+'mhz='+encodeURL(IntToStr(mhz))+'&';
  rep := rep+'ram='+encodeURL(IntToStr(ram))+'&';
  rep := rep+'gigaflops='+encodeURL(IntToStr(GigaFlops))+'&';
  rep := rep+'bits='+encodeURL(IntToStr(bits))+'&';
  rep := rep+'os='+encodeURL(os)+'&';
  rep := rep+'longitude='+encodeURL(FloatToStr(longitude))+'&';
  rep := rep+'latitude='+encodeURL(FloatToStr(latitude))+'&';
  rep := rep+'version='+encodeURL('1.0.0')+'&';
  rep := rep+'team='+encodeURL(team)+'&';
  rep := rep+'userid='+encodeURL(myUserID.userid)+'&';
  rep := rep+'description='+encodeURL(description);//+'&';
end;

 logger_.log(LVL_DEBUG, '[TTransmitClientServiceThread]> Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;


procedure TTransmitClientServiceThread.Execute;
var srv : TServerRecord;
begin
 servMan_.getDefaultServerUrl(srv);
 transmit(srv, '/cluster/report_client.php?'+getReportString, '[TTransmitClientServiceThread]> ', false);
 finishTransmit(srv,  '[TTransmitClientServiceThread]> ', 'Own status transmitted :-)');
end;


end.
