program gpucore;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp,
  { you can add units after this }
  loggers, lockfiles, coreconfigurations,  identities,
  coremodules, servicefactories, servicemanagers,
  servermanagers, dbtablemanagers, coreservices,
  receiveparamservices, receiveserverservices,
  receiveclientservices, transmitclientservices,
  receivechannelservices, coreobjects, coremonitors;

const FRAC_SEC=1/24/3600;

type

  { TGPUCoreApp }

  TGPUCoreApp = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor  Destroy; override;
    procedure   WriteHelp; virtual;
  private
    path_,
    logHeader_   : String;
    coremonitor_ : TCoreMonitor;
    morefrequentupdates_ : TLockFile;

    procedure   mainLoop;
    function    launch(var thread : TCoreServiceThread; tname : String; var srv : TServerRecord) : Boolean;
    procedure   retrieveParamsAndServers;
    procedure   retrieveClients;
    procedure   transmitClient;
    procedure   receiveChannels;
  end;

{ TGPUCoreApp }

procedure TGPUCoreApp.mainLoop;
var tick, days  : Longint;
begin
  coremonitor_.coreStarted;
  logger.logCR; logger.logCR;
  logger.logCR; logger.logCR;
  logger.log(LVL_INFO, logHeader_+'********************');
  logger.log(LVL_INFO, logHeader_+'* Core launched ...*');
  logger.log(LVL_INFO, logHeader_+'********************');
  // main loop
  tick := 1;
  days := 0;
  retrieveParamsAndServers;
  retrieveClients;
  receiveChannels;
  transmitClient;
  while coremonitor_.coreCanRun do
    begin
      if (tick mod 60 = 0) then logger.log(LVL_DEBUG, logHeader_+'Running since '+FloatToStr(myGPUID.Uptime)+' days.');
      if (tick mod myConfID.receive_servers_each = 0) then retrieveParamsAndServers;
      if (tick mod myConfID.receive_nodes_each = 0) then retrieveClients;
      if (tick mod myConfID.transmit_node_each = 0) then transmitClient;
      if (tick mod myConfID.receive_channels_each = 0) then receiveChannels;
      if (tick mod 20 = 0) and morefrequentupdates_.exists then receiveChannels;

      Sleep(1000);

      Inc(tick);
      myGPUID.Uptime := myGPUID.Uptime+FRAC_SEC;

      if (tick>=86400) then
         begin
            tick := 0;
            Inc(days);
         end;
      serviceman.clearFinishedThreads;
    end;

  // last steps
  logger.log(LVL_INFO, logHeader_+'Core was running for '+FloatToStr(myGPUID.uptime)+' days.');
  myGPUID.TotalUptime:=myGPUID.TotalUptime+myGPUID.Uptime;
  myGPUID.Uptime := 0;
  logger.log(LVL_INFO, logHeader_+'Total uptime is '+FloatToStr(myGPUID.TotalUptime)+'.');
  coremonitor_.coreStopped;
end;

function    TGPUCoreApp.launch(var thread : TCoreServiceThread; tname : String; var srv : TServerRecord) : Boolean;
var slot : Longint;
begin
   Result := true;
   logger.log(LVL_DEBUG, logHeader_+tname+' started...');
   slot := serviceman.launch(thread);
   if slot=-1 then
         begin
           Result := false;
           logger.log(LVL_SEVERE, logHeader_+tname+' failed, core too busy!');
         end;

   logger.log(LVL_DEBUG, logHeader_+tname+' over.');
end;


procedure TGPUCoreApp.retrieveParamsAndServers;
var receiveparamthread  : TReceiveParamServiceThread;
    receiveserverthread : TReceiveServerServiceThread;
    srv                 : TServerRecord;
begin
   serverman.getSuperServer(srv);
   receiveparamthread  := servicefactory.createReceiveParamService(srv);
   if not launch(receiveparamthread, 'ReceiveParams', srv) then receiveparamthread.Free;

   receiveserverthread := servicefactory.createReceiveServerService(srv);
   if not launch(receiveserverthread, 'ReceiveServers', srv) then receiveserverthread.Free;
end;

procedure TGPUCoreApp.retrieveClients;
var receiveclientthread  : TReceiveClientServiceThread;
    srv                  : TServerRecord;
begin
   serverman.getServer(srv);
   receiveclientthread  := servicefactory.createReceiveClientService(srv);
   if not launch(receiveclientthread, 'ReceiveClients', srv) then receiveclientthread.Free;
end;

procedure TGPUCoreApp.transmitClient;
var transmitclientthread  : TTransmitClientServiceThread;
    srv                   : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   transmitclientthread  := servicefactory.createTransmitClientService(srv);
   if not launch(transmitclientthread, 'TransmitClient', srv) then transmitclientthread.Free;
end;

procedure TGPUCoreApp.receiveChannels;
var receivechanthread     : TReceiveChannelServiceThread;
    srv                   : TServerRecord;
begin
   serverman.getDefaultServer(srv);
   receivechanthread  := servicefactory.createReceiveChannelService(srv, {srv.chatchannel}'Altos', 'CHAT');
   if not launch(receivechanthread, 'ReceiveChannels', srv) then receivechanthread.Free;
end;

procedure TGPUCoreApp.DoRun;
var
  ErrorMsg : String;
begin
  // quick check parameters
  ErrorMsg:=CheckOptions('h','help');
  if ErrorMsg<>'' then begin
    ShowException(Exception.Create(ErrorMsg));
    Terminate;
    Exit;
  end;

  // parse parameters
  if HasOption('h','help') then begin
    WriteHelp;
    Terminate;
    Exit;
  end;

  mainLoop;

  // stop program loop
  Terminate;
end;


constructor TGPUCoreApp.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
  StopOnException:=True;
  path_ := extractFilePath(ParamStr(0));
  logHeader_ := 'gpucore> ';
  morefrequentupdates_ := TLockFile.Create(path_+PathDelim+'locks', 'morefrequentchat.lock');

  coremonitor_ := TCoreMonitor.Create();
  loadCoreObjects('gpucore');
end;

destructor TGPUCoreApp.Destroy;
begin
  conf.saveCoreConfiguration();
  discardCoreObjects;
  coremonitor_.Free;
  morefrequentupdates_.Free;
  inherited Destroy;
end;

procedure TGPUCoreApp.WriteHelp;
begin
  { add your help code here }
  writeln('Usage: ',ExeName,' -h');
end;

var
  Application: TGPUCoreApp;

{$IFDEF WINDOWS}{$R gpucore.rc}{$ENDIF}

begin
  Application:=TGPUCoreApp.Create(nil);
  Application.Title:='gpucore';
  Application.Run;
  Application.Free;
end.

