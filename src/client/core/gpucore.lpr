program gpucore;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp,
  { you can add units after this }
  loggers, lockfiles,  coreconfigurations,  testconstants, identities,
  coremodules, servicefactories, servicemanagers,
  servermanagers, dbtablemanagers,
  receiveparamservices, receiveserverservices,
  receiveclientservices, transmitclientservices;

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
    logHeader_  : String;
    cms_        : TCoreModule;
    sf_         : TServiceFactory;
    sm_         : TServerManager;
    logger_     : TLogger;
    conf_       : TCoreConfiguration;
    tableman_   : TDbTableManager;
    lock_       : TLockFile;
    serviceman_ : TServiceThreadManager;

    procedure   mainLoop;
    procedure   retrieveParamsAndServers;
    procedure   retrieveClients;
    procedure   transmitClient;
  end;

{ TGPUCoreApp }

procedure TGPUCoreApp.mainLoop;
var tick, days  : Longint;
begin
  logger_.logCR; logger_.logCR;
  logger_.logCR; logger_.logCR;
  logger_.log(LVL_INFO, logHeader_+'********************');
  logger_.log(LVL_INFO, logHeader_+'* Core launched ...*');
  logger_.log(LVL_INFO, logHeader_+'********************');
  // main loop
  tick := 1;
  days := 0;
  retrieveParamsAndServers;
  retrieveClients;
  transmitClient;
  while lock_.exists do
    begin
      if (tick mod 60 = 0) then logger_.log(LVL_DEBUG, logHeader_+'Running since '+FloatToStr(myGPUID.Uptime)+' days.');
      if (tick mod myConfID.receive_servers_each = 0) then retrieveParamsAndServers;
      if (tick mod myConfID.receive_nodes_each = 0) then retrieveClients;
      if (tick mod myConfID.transmit_node_each = 0) then transmitClient;

      Sleep(1000);

      Inc(tick);
      myGPUID.Uptime := myGPUID.Uptime+FRAC_SEC;

      if (tick>=86400) then
         begin
            tick := 0;
            Inc(days);
         end;
      serviceman_.clearFinishedThreads;
    end;

  // last steps
  logger_.log(LVL_INFO, logHeader_+'Core was running for '+FloatToStr(myGPUID.uptime)+' days.');
  myGPUID.TotalUptime:=myGPUID.TotalUptime+myGPUID.Uptime;
  myGPUID.Uptime := 0;
  logger_.log(LVL_INFO, logHeader_+'Total uptime is '+FloatToStr(myGPUID.TotalUptime)+'.');
end;

procedure TGPUCoreApp.retrieveParamsAndServers;
var receiveparamthread  : TReceiveParamServiceThread;
    receiveserverthread : TReceiveServerServiceThread;
    srv                 : TServerRecord;
    slot                : Longint;
begin
   logger_.log(LVL_DEBUG, logHeader_+'RetrieveParamAndServices started...');
   sm_.getSuperServer(srv);
   receiveparamthread  := sf_.createReceiveParamService(srv);
   slot := serviceman_.launch(receiveparamthread);
   if slot=-1 then
          begin
            receiveparamthread.Free;
            logger_.log(LVL_SEVERE, logHeader_+'ReceiveParam failed, core too busy!');
          end;

   receiveserverthread := sf_.createReceiveServerService(srv);
   slot := serviceman_.launch(receiveserverthread);
   if slot=-1 then
         begin
           receiveserverthread.Free;
           logger_.log(LVL_SEVERE, logHeader_+'ReceiveServer failed, core too busy!');
         end;
   logger_.log(LVL_DEBUG, logHeader_+'RetrieveParamAndServices over.');
end;

procedure TGPUCoreApp.retrieveClients;
var receiveclientthread  : TReceiveClientServiceThread;
    srv                  : TServerRecord;
    slot                 : Longint;
begin
   logger_.log(LVL_DEBUG, logHeader_+'RetrieveClients started...');
   sm_.getServer(srv);
   receiveclientthread  := sf_.createReceiveClientService(srv);
   slot := serviceman_.launch(receiveclientthread);
   if slot=-1 then
          begin
            receiveclientthread.Free;
            logger_.log(LVL_SEVERE, logHeader_+'ReceiveClients failed, core too busy!');
          end;

   logger_.log(LVL_DEBUG, logHeader_+'RetrieveClients over.');
end;

procedure TGPUCoreApp.transmitClient;
var transmitclientthread  : TTransmitClientServiceThread;
    srv                   : TServerRecord;
    slot                  : Longint;
begin
   logger_.log(LVL_DEBUG, logHeader_+'TransmitClient started...');
   sm_.getDefaultServer(srv);
   transmitclientthread  := sf_.createTransmitClientService(srv);
   slot := serviceman_.launch(transmitclientthread);
   if slot=-1 then
          begin
            transmitclientthread.Free;
            logger_.log(LVL_SEVERE, logHeader_+'TransmitClient failed, core too busy!');
          end;

   logger_.log(LVL_DEBUG, logHeader_+'TransmitClient over.');
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

  lock_     := TLockFile.Create(path_+PathDelim+'locks', 'coreapp.lock');
  logger_   := TLogger.Create(path_+PathDelim+'logs', 'coreapp.log', 'coreapp.old', LVL_DEBUG, 1024*1024);
  conf_     := TCoreConfiguration.Create(path_, 'coreapp.ini');
  conf_.loadConfiguration();
  tableman_ := TDbTableManager.Create(path_+PathDelim+'coreapp-db.sqlite');
  tableman_.OpenAll;
  sm_          := TServerManager.Create(conf_, tableman_.getServerTable(), logger_);
  cms_         := TCoreModule.Create(logger_, path_, 'dll');
  sf_          := TServiceFactory.Create(sm_, tableman_, PROXY_HOST, PROXY_PORT, logger_, conf_);
  serviceman_  := TServiceThreadManager.Create(tmServiceStatus.maxthreads);
end;

destructor TGPUCoreApp.Destroy;
begin
  conf_.saveConfiguration();

  serviceman_.Free;
  cms_.Free;
  sf_.Free;
  sm_.Free;
  tableman_.CloseAll;
  tableman_.Free;
  conf_.Free;
  logger_.Free;
  lock_.Free;
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

