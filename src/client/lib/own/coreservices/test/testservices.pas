unit testservices;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  servicemanagers, servicefactories, servermanagers, testconstants,
  loggers, dbtablemanagers, receivenodeservices, transmitnodeservices,
  coreconfigurations;

type

  TTestServices = class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestReceiveNodeService;
    procedure TestTransmitNodeService;

  private
    serviceMan_  : TServiceThreadManager;
    srvFactory_  : TServiceFactory;
    serverMan_   : TServerManager;
    tableMan_    : TDbTableManager;
    conf_        : TCoreConfiguration;

    path_          : String;
    logger_        : TLogger;
    urls_          : TStringList;

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

procedure TTestServices.TestReceiveNodeService;
var rcvnodeThread : TReceiveNodeServiceThread;
begin
  rcvnodeThread := srvFactory_.createReceiveNodeService();
  serviceMan_.launch(rcvnodeThread);
  waitForCompletion();
end;

procedure TTestServices.TestTransmitNodeService;
var trxnodeThread : TTransmitNodeServiceThread;
begin
  trxnodeThread := srvFactory_.createTransmitNodeService();
  serviceMan_.launch(trxnodeThread);
  waitForCompletion();
end;

procedure TTestServices.SetUp;
begin
  path_           := ExtractFilePath(ParamStr(0));
  logger_         := TLogger.Create(path_+'logs', 'services.log');
  logger_.setLogLevel(LVL_DEBUG);
  urls_           := TStringList.Create;
  urls_.add('http://www.gpu-grid.net/file_distributor');
  serverMan_      := TServerManager.Create(urls_, 0);
  tableMan_       := TDbTableManager.Create(path_+PathDelim+'core.db');
  tableMan_.openAll();
  conf_           := TCoreConfiguration.Create(path_, 'core.ini');
  conf_.loadConfiguration();

  serviceMan_  := TServiceThreadManager.Create(3);
  srvFactory_  := TServiceFactory.Create(serverMan_, tableMan_, PROXY_HOST, PROXY_PORT, logger_, conf_);
end; 

procedure TTestServices.TearDown;
begin
 conf_.Free;
 tableMan_.closeAll();
 serviceMan_.Free;
 srvFactory_.Free;
 tableMan_.Free;
 serverMan_.Free;
 logger_.Free;
 urls_.Free;
end; 

initialization

  RegisterTest(TTestServices);
end.

