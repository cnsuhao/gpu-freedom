unit transmitnodeservices;

interface

uses coreconfigurations, coreservices, downloadutils, httpsend;


type TTransmitNodeServiceThread = class(TTransmitServiceThread)
 public
  constructor Create(servMan : TServerManager; proxy, port : String; logger : TLogger
                     conf : TCoreConfiguration);
 protected
  procedure Execute; override;

 private
    conf_ : TCoreConfiguration;
    function getReportString : String;
end;



implementation

constructor TTransmitNodeServiceThread.Create(servMan : TServerManager; proxy, port : String; logger : TLogger
            conf : TCoreConfiguration);
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
 rep := 'processor='+URLEncode(gid.nodename)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';
 rep := rep+''+URLEncode(gid.)+'&';

 logger_.log(LVL_DEBUG, '[TTransmitNodeServiceThread]> Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;


procedure TTransmitNodeServiceThread.Execute; override;
var
    stream    : TMemoryStream;
    proxyseed : String;
begin
 stream  := TMemoryStream.Create;
 proxyseed  := getProxySeed;
 erroneous_ := not downloadToStream(servMan_.getServerUrl()+'/report_node_info.php?randomseed='+proxyseed+'&'+getReportString,
               proxy_, port_, '[TTransmitNodeServiceThread]> ', logger_, stream);

 if stream <>nil then stream.Free  else logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Internal error in receivenodeservices.pas, stream is nil');
 done_ := true;
end;


end.
