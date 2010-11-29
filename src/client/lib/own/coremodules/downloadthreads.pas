unit downloadthreads;
{
  DownloadThread is a thread which downloads a file
  via HTTP and stores it in targetfile under directory
  targetDir.

  This file is build with testhttp as template
   which is part of the Synapse library
  available under /src/client/lib/ext/synapse.

  (c) by 2002-2010 the GPU Development Team
  This unit is released under GNU Public License (GPL)
}

interface

uses
  managedthreads, downloadutils, loggers, sysutils;


type TDownloadThread = class(TManagedThread)
 public
   constructor Create(url, targetPath, targetFilename, proxy, port : String; var logger : TLogger);
   function    getTargetFileName() : String;

 protected
    procedure Execute; override;

 private
    url_,
    targetPath_,
    targetFile_,
    proxy_,
    port_       : String;
    logger_     : TLogger;

end;


implementation


constructor TDownloadThread.Create(url, targetPath, targetFilename, proxy, port : String; var logger : TLogger);
begin
  inherited Create();

  logger_ := logger;
  url_ := url;
  targetPath_ := targetPath;
  targetFile_ := targetFileName;
  proxy_ := proxy;
  port_ := port;
end;

function  TDownloadThread.getTargetFileName() : String;
begin
  Result := targetPath_+PathDelim+targetFile_;
end;

procedure TDownloadThread.execute();
var AltFilename : String;
    index       : Longint;
begin
  if FileExists(targetPath_+targetFile_) then
  begin
    index := 2;
    repeat
      AltFileName := targetFile_ + '.' + IntToStr(index);
      inc(index);
    until not FileExists(targetPath_+AltFileName);
    logger_.log(LVL_WARNING, '"'+targetFile_+'" exists, writing to "'+AltFileName+'"');
    targetFile_ := AltFileName;
  end;

  erroneous_ := not downloadToFile(url_, targetPath_, targetFile_,
                                   proxy_, port_,
                                   'DownloadThread ['+targetFile_+']> ', logger_);
   done_ := true;
end;

end.
