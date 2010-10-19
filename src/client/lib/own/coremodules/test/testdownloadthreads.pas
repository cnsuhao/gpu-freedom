unit testdownloadthreads;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  downloadthreads, loggers;

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
  if FileExists(path+'index.php') then DeleteFile(path+'index.php');

  downThread_ := TDownloadThread.Create('http://www.gpu-grid.net', path, 'index.php', logger_);

  while not downThread_.isDone() do Sleep(100);

  AssertEquals('File index.php exists', true, FileExists(path+'index.php'));
  downThread_.Free;
  logger_.Free;
end;


initialization

  RegisterTest(TTestDownloadThread); 
end.

