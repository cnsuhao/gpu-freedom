unit testdownloadthreadmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  testconstants, downloadthreadmanagers, loggers;

type

  TTestDownloadThreadManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestDownThreadManager;
  private
    logger_      : TLogger;
    downManager_ : TDownloadThreadManager;
    path_        : String;
  end;

implementation

procedure TTestDownloadThreadManager.TestDownThreadManager;
begin
 downManager_.setMaxThreads(3);
 if FileExists(path_+'online.xml') then DeleteFile(path_+'online.xml');
 if FileExists(path_+'deltasql.php') then DeleteFile(path_+'deltasql.php');
 if FileExists(path_+'gpu.php') then DeleteFile(path_+'gpu.php');
 downManager_.download('http://www.gpu-grid.net/file_distributor/list_computers_online_xml.php', path_, 'online.xml');
 downManager_.download('http://www.deltasql.org/index.php', path_, 'deltasql.php');
 downManager_.download('http://gpu.sourceforge.net/index.php', path_, 'gpu.php');
 while not downManager_.isIdle do
    begin
      downManager_.clearFinishedThreads;
      sleep(100);
    end;
 AssertEquals('online.xml exists', true, FileExists(path_+'online.xml'));
 AssertEquals('deltasql.php exists', true, FileExists(path_+'deltasql.php'));
 AssertEquals('gpu.php exists', true, FileExists(path_+'gpu.php'));
end;

procedure TTestDownloadThreadManager.SetUp; 
begin
  path_           := ExtractFilePath(ParamStr(0));
  logger_         := TLogger.Create(path_+'logs', 'core.log');
  logger_.setLogLevel(LVL_DEBUG);

  downManager_ := TDownloadThreadManager.Create(logger_);
  downManager_.setProxy(PROXY_HOST, PROXY_PORT);
end; 

procedure TTestDownloadThreadManager.TearDown; 
begin
  downManager_.Free;
  logger_.Free;
end; 

initialization

  RegisterTest(TTestDownloadThreadManager); 
end.

