unit transmitnodeservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
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
    cid : TConfIdentity;
begin
 uid := conf_.getUserIdentity;
 gid := conf_.getGPUIdentity;
 cid := conf_.getConfIdentity;
 rep :=     'nodename='+encodeURL(gid.nodename)+'&';
 rep := rep+'nodeid='+encodeURL(gid.nodeid)+'&';
 rep := rep+'country='+encodeURL(gid.country)+'&';
 rep := rep+'region='+encodeURL(gid.region)+'&';
 rep := rep+'city='+encodeURL(gid.city)+'&';
 rep := rep+'zip='+encodeURL(gid.zip)+'&';
 rep := rep+'uptime='+encodeURL(FloatToStr(gid.uptime))+'&';
 rep := rep+'totaluptime='+encodeURL(FloatToStr(gid.totaluptime))+'&';
 rep := rep+'ip='+encodeURL(gid.ip)+'&';
 rep := rep+'localip='+encodeURL(gid.localip)+'&';
 rep := rep+'port='+encodeURL(gid.port)+'&';
 rep := rep+'acceptincoming='+encodeURL('false')+'&';
 rep := rep+'cputype='+encodeURL(gid.cputype)+'&';
 rep := rep+'mhz='+encodeURL(IntToStr(gid.mhz))+'&';
 rep := rep+'ram='+encodeURL(IntToStr(gid.ram))+'&';
 rep := rep+'gigaflops='+encodeURL(IntToStr(gid.GigaFlops))+'&';
 rep := rep+'bits='+encodeURL(IntToStr(gid.bits))+'&';
 rep := rep+'os='+encodeURL(gid.os)+'&';
 rep := rep+'longitude='+encodeURL(FloatToStr(gid.longitude))+'&';
 rep := rep+'latitude='+encodeURL(FloatToStr(gid.latitude))+'&';
 rep := rep+'version='+encodeURL('1.0.0')+'&';
 rep := rep+'team='+encodeURL(gid.team)+'&';
 rep := rep+'userid='+encodeURL(uid.userid)+'&';
 rep := rep+'defaultservername='+encodeURL(cid.default_server_name)+'&';

 logger_.log(LVL_DEBUG, '[TTransmitNodeServiceThread]> Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;


procedure TTransmitNodeServiceThread.Execute;
begin
 transmit(servMan_.getDefaultServerUrl(),'/report_nodeinfo.php?'+getReportString, '[TTransmitNodeServiceThread]> ', false);
 finishComm(servMan_.getDefaultServerUrl(),  '[TTransmitNodeServiceThread]> ', 'Own status transmitted :-)');
end;


end.
