unit testservices;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  servicemanagers, servicefactories, servermanagers, testconstants,
  loggers, dbtablemanagers;

type

  TTestServices = class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestReceiveService;

  private
    serviceMan_  : TServiceThreadManager;
    srvFactory_  : TServiceFactory;
    serverMan_   : TServerManager;
    tableMan_    : TDbTableManager;

    path_        : String;
    logger_      : TLogger;
    urls_        : TStringList;

  end; 

implementation

procedure TTestServices.TestReceiveService;
begin


end; 

procedure TTestServices.SetUp;
begin
  path_           := ExtractFilePath(ParamStr(0));
  logger_         := TLogger.Create(path_+'logs', 'services.log');
  logger_.setLogLevel(LVL_DEBUG);
  urls_           := TStringList.Create;
  urls_.add('http://www.gpu-grid.net/file_distributor');
  serverMan_      := TServerManager.Create(urls_, 0);
  tableMan_       := TDbTableManager.Create(path_+'\core.db');

  serviceMan_  := TServiceThreadManager.Create(3);
  srvFactory_  := TServiceFactory.Create(serverMan_, tableMan_, PROXY_HOST, PROXY_PORT, logger_);
end; 

procedure TTestServices.TearDown;
begin
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

