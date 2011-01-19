unit transmitclientservices;

interface

uses coreconfigurations, coreservices, synacode, stkconstants,
     clienttables, servermanagers, loggers, identities, dbtablemanagers,
     SysUtils, Classes;


type TTransmitClientServiceThread = class(TTransmitServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);
 protected
  procedure Execute; override;

 private
    function  getPHPArguments() : AnsiString;
    procedure insertTransmission();
end;



implementation

constructor TTransmitClientServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TTransmitClientServiceThread]> ', conf, tableman);
end;

function TTransmitClientServiceThread.getPHPArguments() : AnsiString;
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
  rep := rep+'uptime='+encodeURL(FloatToStr(uptime))+'&';           //TODO: FloatToStr with formatset
  rep := rep+'totaluptime='+encodeURL(FloatToStr(totaluptime))+'&'; //TODO: FloatToStr with formatset
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
  rep := rep+'longitude='+encodeURL(FloatToStr(longitude))+'&';  //TODO: FloatToStr with formatset
  rep := rep+'latitude='+encodeURL(FloatToStr(latitude))+'&';    //TODO: FloatToStr with formatset
  rep := rep+'version='+encodeURL('1.0.0')+'&';
  rep := rep+'team='+encodeURL(team)+'&';
  rep := rep+'userid='+encodeURL(myUserID.userid)+'&';
  rep := rep+'description='+encodeURL(description);//+'&';
end;

 logger_.log(LVL_DEBUG, logHeader_+'Reporting string is:');
 logger_.log(LVL_DEBUG, rep);
 Result := rep;
end;

procedure TTransmitClientServiceThread.insertTransmission();
var row : TDbClientRow;
begin
 with myGPUID do
 begin
   row.nodeid            := nodeid;
   row.server_id         := srv_.id;
   row.nodename          := nodename;
   row.country           := country;
   row.region            := region;
   row.city              := city;
   row.zip               := zip;
   row.description       := description;
   row.ip                := ip;
   row.port              := port;
   row.localip           := localip;
   row.os                := os;
   row.cputype           := cputype;
   row.version           := version;
   row.acceptincoming    := acceptincoming;
   row.gigaflops    := gigaflops;
   row.ram          := ram;
   row.mhz          := mhz;
   row.nbcpus       := nbcpus;
   row.bits         := bits;
   row.online  := true;
   row.updated := true;
   row.uptime      := uptime;
   row.totaluptime := totaluptime;
   row.longitude   := longitude;
   row.latitude    := latitude;
   row.userid      := myUserId.userid;
   row.team        := team;
  end;

 tableman_.getClientTable().insertOrUpdate(row);
 logger_.log(LVL_DEBUG, logHeader_+'Updated or added <'+row.nodename+'> to tbclient table.');
end;

procedure TTransmitClientServiceThread.Execute;
begin
 insertTransmission();
 transmit('/cluster/report_client.php?'+getPHPArguments(), false);
 finishTransmit('Own status transmitted :-)');
end;


end.
