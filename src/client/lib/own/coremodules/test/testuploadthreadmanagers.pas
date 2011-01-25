unit testuploadthreadmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  testconstants, uploadthreadmanagers, loggers;

type

  TTestUploadThreadManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestUpThreadManager;
  private
    logger_      : TLogger;
    upManager_   : TUploadThreadManager;
    path_        : String;
  end;

implementation

procedure TTestUploadThreadManager.TestUpThreadManager;
begin
 upManager_.setMaxThreads(3);
 AssertEquals('online.xml exists', true, FileExists(path_+'online.xml'));
 AssertEquals('deltasql.php exists', true, FileExists(path_+'deltasql.php'));
 AssertEquals('gpu.php exists', true, FileExists(path_+'gpu.php'));

 upManager_.upload('http://www.gpu-grid.net/superserver/test/http_upload_test.php', path_, 'online.xml');
 upManager_.upload('http://www.gpu-grid.net/superserver/test/http_upload_test.php', path_, 'deltasql.php');
 upManager_.upload('http://www.gpu-grid.net/superserver/test/http_upload_test.php', path_, 'gpu.php');
 while not upManager_.isIdle do
    begin
      upManager_.clearFinishedThreads;
      sleep(100);
    end;

 if FileExists(path_+'online.xml') then DeleteFile(path_+'online.xml');
 if FileExists(path_+'deltasql.php') then DeleteFile(path_+'deltasql.php');
 if FileExists(path_+'gpu.php') then DeleteFile(path_+'gpu.php');
end;

procedure TTestUploadThreadManager.SetUp;
begin
  path_           := ExtractFilePath(ParamStr(0));
  logger_         := TLogger.Create(path_+'logs', 'core.log');
  logger_.setLogLevel(LVL_DEBUG);

  upManager_ := TUploadThreadManager.Create(logger_);
  upManager_.setProxy(PROXY_HOST, PROXY_PORT);
end; 

procedure TTestUploadThreadManager.TearDown; 
begin
  upManager_.Free;
  logger_.Free;
end; 

initialization

  RegisterTest(TTestUploadThreadManager); 
end.

