unit testservices;

{$mode objfpc}{$H+}

interface

uses
   Classes, SysUtils, fpcunit, testutils, testregistry,
  servicemanagers, servicefactories, servermanagers, testconstants,
  loggers, dbtablemanagers, receiveclientservices, transmitclientservices,
  receiveserverservices, receiveparamservices,
  receivechannelservices, transmitchannelservices,
  receivejobservices, transmitjobservices, jobdefinitiontables,
  receivejobresultservices, transmitjobresultservices, jobresulttables,
  coreconfigurations, coreservices, workflowmanagers, coremodules;

type

  TTestServices = class(TTestCase)

  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestReceiveParamService;
    procedure TestReceiveServerService;
    procedure TestTransmitClientService;
    procedure TestReceiveClientService;
    procedure TestReceiveChannelService;
    procedure TestTransmitChannelService;
    procedure TestReceiveJobService;
    procedure TestReceiveJobResultService;


  private
    serviceMan_  : TServiceThreadManager;
    srvFactory_  : TServiceFactory;
    serverMan_   : TServerManager;
    tableMan_    : TDbTableManager;
    conf_        : TCoreConfiguration;
    workflowMan_ : TWorkflowManager;

    path_          : String;
    logger_        : TLogger;
    coremodule_    : TCoreModule;

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
    srv            : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  rcvparamThread := srvFactory_.createReceiveParamService(srv);
  serviceMan_.launch(TCoreServiceThread(rcvparamThread), 'ReceiveParamThread');
  waitForCompletion();
end;


procedure TTestServices.TestReceiveServerService;
var rcvserverThread : TReceiveServerServiceThread;
    srv             : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  rcvserverThread := srvFactory_.createReceiveServerService(srv);
  serviceMan_.launch(TCoreServiceThread(rcvserverThread), 'ReceiveServerThread');
  waitForCompletion();
end;

procedure TTestServices.TestTransmitClientService;
var trxclientThread : TTransmitClientServiceThread;
    srv             : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  trxclientThread := srvFactory_.createTransmitClientService(srv);
  serviceMan_.launch(TCoreServiceThread(trxclientThread), 'TransmitClientThread');
  waitForCompletion();
end;

procedure TTestServices.TestReceiveClientService;
var rcvclientThread : TReceiveClientServiceThread;
    srv             : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  rcvclientThread := srvFactory_.createReceiveClientService(srv);
  serviceMan_.launch(TCoreServiceThread(rcvclientThread), 'ReceiveClientThread');
  waitForCompletion();
end;

procedure TTestServices.TestReceiveChannelService;
var thread : TReceiveChannelServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createReceiveChannelService(srv, 'Altos', 'CHAT');
  serviceMan_.launch(TCoreServiceThread(thread), 'ReceiveChannelThread');
  waitForCompletion();
end;

procedure TTestServices.TestTransmitChannelService;
var thread : TTransmitChannelServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createTransmitChannelService(srv, 'Altos', 'CHAT', 'hello world :-)');
  serviceMan_.launch(TCoreServiceThread(thread), 'TransmitChannelThread');
  waitForCompletion();
end;

procedure TTestServices.TestReceiveJobService;
var thread : TReceiveJobServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createReceiveJobService(srv);
  serviceMan_.launch(TCoreServiceThread(thread), 'ReceiveJobServiceThread');
  waitForCompletion();
end;


procedure TTestServices.TestReceiveJobResultService;
var thread : TReceiveJobResultServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createReceiveJobResultService(srv, '12345');
  serviceMan_.launch(TCoreServiceThread(thread), 'ReceiveJobResultService');
  waitForCompletion();
end;


procedure TTestServices.SetUp;
begin
  path_           := ExtractFilePath(ParamStr(0));

  if not DirectoryExists(path_+'logs') then
     CreateDir(path_+'logs');
  logger_         := TLogger.Create(path_+'logs', 'coreservicestest.log');
  logger_.setLogLevel(LVL_DEBUG);

  tableMan_       := TDbTableManager.Create(path_+'coreservicestest.db');
  tableMan_.openAll();
  conf_           := TCoreConfiguration.Create(path_);
  conf_.loadConfiguration();
  serverMan_      := TServerManager.Create(conf_,
                                           tableMan_.getServerTable(),
                                           logger_);

  serviceMan_  := TServiceThreadManager.Create(3);
  coremodule_  := TCoreModule.Create(logger_, path_, 'dll');
  workflowMan_ := TWorkflowManager.Create(tableman_, logger_);
  srvFactory_  := TServiceFactory.Create(workflowMan_, serverMan_, tableMan_, PROXY_HOST, PROXY_PORT, logger_, conf_, coreModule_);
end;

procedure TTestServices.TearDown;
begin
 tableMan_.closeAll();
 coreModule_.Free;
 workflowMan_.Free;
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

