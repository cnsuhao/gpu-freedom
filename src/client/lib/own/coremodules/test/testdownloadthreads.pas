unit testdownloadthreads;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  downloadthreads, loggers, testconstants;

type

  TTestDownloadThread= class(TTestCase)
  published
    procedure TestDownloadThread;
  private
    logger_     : TLogger;
    downThread_ : TDownloadThread;
  end; 

implementation

procedure TTestDownloadThread.TestDownloadThread;
var path : String;
begin
  path            := ExtractFilePath(ParamStr(0));
  logger_         := TLogger.Create(path+PathDelim+'logs', 'core.log');
  logger_.setLogLevel(LVL_DEBUG);
  if FileExists(path+'index.html') then DeleteFile(path+'index.html');

  downThread_ := TDownloadThread.Create('http://www.google.ch/index.html', path, 'index.html', PROXY_HOST, PROXY_PORT, logger_, true);

  while not downThread_.isDone() do Sleep(100);

  AssertEquals('File index.html exists', true, FileExists(path+'index.html'));
  downThread_.Free;
  logger_.Free;
end;


initialization

  RegisterTest(TTestDownloadThread); 
end.

