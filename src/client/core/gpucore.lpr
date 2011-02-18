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
  receiveparamservices, receiveserverservices;

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
    path_       : String;
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
  end;

{ TGPUCoreApp }

procedure TGPUCoreApp.mainLoop;
var tick : Longint;
begin
  // main loop
  tick := 1;
  RetrieveParamsAndServers;
  while lock_.exists do
    begin
      if (tick mod myConfID.receive_servers_each = 0) then
               retrieveParamsAndServers;

      Sleep(1000);

      Inc(tick);
      if (tick>=86400) then tick := 0;
      serviceman_.clearFinishedThreads;
    end;
end;

procedure TGPUCoreApp.retrieveParamsAndServers;
var receiveparamthread  : TReceiveParamServiceThread;
    receiveserverthread : TReceiveServerServiceThread;
    srv                 : TServerRecord;
    slot                : Longint;
begin
   sm_.getSuperServer(srv);
   receiveparamthread  := sf_.createReceiveParamService(srv);
   slot := serviceman_.launch(receiveparamthread);
   if slot=-1 then receiveparamthread.Free;

   receiveserverthread := sf_.createReceiveServerService(srv);
   slot := serviceman_.launch(receiveserverthread);
   if slot=-1 then receiveserverthread.Free;
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

  lock_     := TLockFile.Create(path_+PathDelim+'locks', 'coreapp.lock');
  logger_   := TLogger.Create(path_+PathDelim+'logs', 'coreapp.log', 'coreapp.old', LVL_DEFAULT, 1024*1024);
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

