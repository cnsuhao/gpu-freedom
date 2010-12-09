unit transmitnodeservices;

interface

uses coreconfigurations, coreservices, downloadutils, synacode, stkconstants,
     servermanagers, loggers, identities, SysUtils, Classes;


type TTransmitNodeServiceThread = class(TTransmitServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration);
 protected
  procedure Execute; override;

 private
    conf_ : TCoreConfiguration;
    function getReportString : String;
end;



implementation

constructor TTransmitNodeServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                                              var conf : TCoreConfiguration);
begin
 inherited Create(servMan, proxy, port, logger);
 conf_ := conf;
end;

function TTransmitNodeServiceThread.getReportString : String;
var rep : String;
    uid : TUserIdentity;
    gid : TGPUIdentity;
begin
 uid := conf_.getUserIdentity;
 gid := conf_.getGPUIdentity;
 rep :=     'processor='+encodeURL(gid.nodename)+'&';
 rep := rep+'nodeid='+encodeURL(gid.nodeid)+'&';
 rep := rep+'country='+encodeURL(gid.country)+'&';
 rep := rep+'uptime='+encodeURL(FloatToStr(gid.uptime))+'&';
 rep := rep+'totuptime='+encodeURL(FloatToStr(gid.totaluptime))+'&';
 rep := rep+'ip='+encodeURL(gid.ip)+'&';
 rep := rep+'port='+encodeURL(IntToStr(gid.port))+'&';
 rep := rep+'acceptincoming='+encodeURL('false')+'&';
 rep := rep+'cputype='+encodeURL(gid.cputype)+'&';
 rep := rep+'mhz='+encodeURL(IntToStr(gid.mhz))+'&';
 rep := rep+'ram='+encodeURL(IntToStr(gid.ram))+'&';
 rep := rep+'os='+encodeURL(gid.os)+'&';
 rep := rep+'freeconn='+encodeURL('0')+'&';
 rep := rep+'maxconn='+encodeURL('0')+'&';
 rep := rep+'lon='+encodeURL(FloatToStr(gid.longitude))+'&';
 rep := rep+'lat='+encodeURL(FloatToStr(gid.latitude))+'&';
 rep := rep+'version='+encodeURL('1.000')+'&';
 rep := rep+'team='+encodeURL(gid.team)+'&';

 logger_.log(LVL_DEBUG, '[TTransmitNodeServiceThread]> Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;


procedure TTransmitNodeServiceThread.Execute;
var
    stream    : TMemoryStream;
    proxyseed : String;
begin
 stream  := TMemoryStream.Create;
 proxyseed  := getProxySeed;
 erroneous_ := not downloadToStream(servMan_.getDefaultServerUrl()+'/report_nodeinfo.php?randomseed='+proxyseed+'&'+getReportString,
               proxy_, port_, '[TTransmitNodeServiceThread]> ', logger_, stream);

 if stream <>nil then stream.Free  else logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Internal error in receivenodeservices.pas, stream is nil');
 if erroneous_ then
   logger_.log(LVL_SEVERE, '[TTransmitNodeServiceThread]> Thread finished but ERRONEOUS flag set :-(')
 else
   logger_.log(LVL_INFO, '[TTransmitNodeServiceThread]> Our status transmitted to default server succesfully :-)');
 done_ := true;
end;


end.
