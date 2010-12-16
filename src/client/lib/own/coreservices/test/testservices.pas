unit testservices;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  servicemanagers, servicefactories, servermanagers, testconstants,
  loggers, dbtablemanagers, receiveclientservices, transmitclientservices,
  receiveserverservices, receiveparamservices, coreconfigurations;

type

  TTestServices = class(TTestCase)

  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestReceiveParamService;
    procedure TestReceiveServerService;
    procedure TestReceiveClientService;
    procedure TestTransmitClientService;

  private
    serviceMan_  : TServiceThreadManager;
    srvFactory_  : TServiceFactory;
    serverMan_   : TServerManager;
    tableMan_    : TDbTableManager;
    conf_        : TCoreConfiguration;

    path_          : String;
    logger_        : TLogger;

    procedure waitForCompletion();
  end; 

implementation


procedure TTestServices.waitForCompletion();
begin
  while not serviceMan_.isIdle() do
      begin
        Sleep(150);
        serviceMan_.clearFinishedThreads();
      end;
end;

procedure TTestServices.TestReceiveParamService;
var rcvparamThread : TReceiveParamServiceThread;
begin
  rcvparamThread := srvFactory_.createReceiveParamService();
  serviceMan_.launch(rcvparamThread);
  waitForCompletion();
end;


procedure TTestServices.TestReceiveServerService;
var rcvserverThread : TReceiveServerServiceThread;
begin
  rcvserverThread := srvFactory_.createReceiveServerService();
  serviceMan_.launch(rcvserverThread);
  waitForCompletion();
end;


procedure TTestServices.TestReceiveClientService;
var rcvclientThread : TReceiveClientServiceThread;
begin
  rcvclientThread := srvFactory_.createReceiveClientService();
  serviceMan_.launch(rcvclientThread);
  waitForCompletion();
end;

procedure TTestServices.TestTransmitClientService;
var trxclientThread : TTransmitClientServiceThread;
begin
  trxclientThread := srvFactory_.createTransmitClientService();
  serviceMan_.launch(trxclientThread);
  waitForCompletion();
end;

procedure TTestServices.SetUp;
begin
  path_           := ExtractFilePath(ParamStr(0));

  if not DirectoryExists(path_+'logs') then
     CreateDir(path_+'logs');
  logger_         := TLogger.Create(path_+'logs', 'services.log');
  logger_.setLogLevel(LVL_DEBUG);

  tableMan_       := TDbTableManager.Create(path_+'core.db');
  tableMan_.openAll();
  conf_           := TCoreConfiguration.Create(path_, 'core.ini');
  conf_.loadConfiguration();
  serverMan_      := TServerManager.Create(conf_,
                                           tableMan_.getServerTable(),
                                           logger_);

  serviceMan_  := TServiceThreadManager.Create(3);
  srvFactory_  := TServiceFactory.Create(serverMan_, tableMan_, PROXY_HOST, PROXY_PORT, logger_, conf_);
end;

procedure TTestServices.TearDown;
begin
 tableMan_.closeAll();
 serviceMan_.Free;
 srvFactory_.Free;
 tableMan_.Free;
 serverMan_.Free;
 conf_.Free;
 logger_.Free;
end; 

initialization

  RegisterTest(TTestServices);
end.

