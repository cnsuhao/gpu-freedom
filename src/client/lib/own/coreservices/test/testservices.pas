unit testservices;

{$mode objfpc}{$H+}

interface

uses
   Classes, SysUtils, fpcunit, testutils, testregistry,
  servicemanagers, servicefactories, servermanagers, testconstants,
  loggers, dbtablemanagers, receiveclientservices, transmitclientservices,
  receiveserverservices, receiveparamservices,
  receivechannelservices, transmitchannelservices,
  receivejobservices, transmitjobservices, jobtables,
  receivejobresultservices, transmitjobresultservices, jobresulttables,
  coreconfigurations;

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
    procedure TestTransmitJobService;
    procedure TestReceiveJobResultService;
    procedure TestTransmitJobResultService;


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
    srv            : TServerRecord;
begin
  serverMan_.getSuperServer(srv);
  rcvparamThread := srvFactory_.createReceiveParamService(srv);
  serviceMan_.launch(rcvparamThread);
  waitForCompletion();
end;


procedure TTestServices.TestReceiveServerService;
var rcvserverThread : TReceiveServerServiceThread;
    srv             : TServerRecord;
begin
  serverMan_.getSuperServer(srv);
  rcvserverThread := srvFactory_.createReceiveServerService(srv);
  serviceMan_.launch(rcvserverThread);
  waitForCompletion();
end;

procedure TTestServices.TestTransmitClientService;
var trxclientThread : TTransmitClientServiceThread;
    srv             : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  trxclientThread := srvFactory_.createTransmitClientService(srv);
  serviceMan_.launch(trxclientThread);
  waitForCompletion();
end;

procedure TTestServices.TestReceiveClientService;
var rcvclientThread : TReceiveClientServiceThread;
    srv             : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  rcvclientThread := srvFactory_.createReceiveClientService(srv);
  serviceMan_.launch(rcvclientThread);
  waitForCompletion();
end;

procedure TTestServices.TestReceiveChannelService;
var thread : TReceiveChannelServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getSuperServer(srv);
  thread := srvFactory_.createReceiveChannelService(srv, 'Altos', 'CHAT');
  serviceMan_.launch(thread);
  waitForCompletion();
end;

procedure TTestServices.TestTransmitChannelService;
var thread : TTransmitChannelServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getSuperServer(srv);
  thread := srvFactory_.createTransmitChannelService(srv, 'Altos', 'CHAT', 'hello world :-)');
  serviceMan_.launch(thread);
  waitForCompletion();
end;

procedure TTestServices.TestReceiveJobService;
var thread : TReceiveJobServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createReceiveJobService(srv);
  serviceMan_.launch(thread);
  waitForCompletion();
end;

procedure TTestServices.TestTransmitJobService;
var thread : TTransmitJobServiceThread;
    srv    : TServerRecord;
    jobrow : TDbJobRow;
begin
  jobrow.job := '1, 1, add';
  jobrow.jobid := '12345';
  jobrow.islocal:=false;
  jobrow.requests:=3;
  jobrow.status:=JS_NEW;
  jobrow.workunitincoming:='pari.txt';
  jobrow.workunitoutgoing:='para.txt';
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createTransmitJobService(srv, jobrow);
  serviceMan_.launch(thread);
  waitForCompletion();
end;

procedure TTestServices.TestReceiveJobResultService;
var thread : TReceiveJobResultServiceThread;
    srv    : TServerRecord;
begin
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createReceiveJobResultService(srv, '12345');
  serviceMan_.launch(thread);
  waitForCompletion();
end;


procedure TTestServices.TestTransmitJobResultService;
var thread : TTransmitJobResultServiceThread;
    srv    : TServerRecord;
    jobresultrow : TDbJobResultRow;
begin
  jobresultrow.jobresult := '2';
  jobresultrow.jobid := '12345';
  jobresultrow.requestid := 1;
  jobresultrow.workunitresult:='';
  jobresultrow.iserroneous := false;
  jobresultrow.errorid := 0;
  jobresultrow.errormsg := '';
  jobresultrow.errorarg := '';
  jobresultrow.nodeid := '123';
  jobresultrow.nodename := 'hola';
  jobresultrow.job_id := -1;
  serverMan_.getDefaultServer(srv);
  thread := srvFactory_.createTransmitJobResultService(srv, jobresultrow);
  serviceMan_.launch(thread);
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
  conf_           := TCoreConfiguration.Create(path_);
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

