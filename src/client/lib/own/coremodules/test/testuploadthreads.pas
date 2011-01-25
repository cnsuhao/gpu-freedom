unit testuploadthreads;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  uploadthreads, loggers, testconstants;

type

  TTestUploadThread= class(TTestCase)
  published
    procedure TestUploadThread;
  private
    logger_     : TLogger;
    upThread_   : TUploadThread;
  end; 

implementation

procedure TTestUploadThread.TestUploadThread;
var path : String;
begin
  path            := ExtractFilePath(ParamStr(0));
  logger_         := TLogger.Create(path+PathDelim+'logs', 'core.log');
  logger_.setLogLevel(LVL_DEBUG);
  AssertEquals('File online.xml exists for upload', true, FileExists(path+'online.xml'));

  upThread_ := TUploadThread.Create('http://www.gpu-grid.net/superserver/test/http_upload_test.php', path, 'online.xml', PROXY_HOST, PROXY_PORT, logger_);

  while not upThread_.isDone() do Sleep(100);

  upThread_.Free;
  logger_.Free;
end;


initialization

  RegisterTest(TTestUploadThread); 
end.

