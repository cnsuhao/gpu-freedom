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
  sysutils, strutils, httpsend, Classes,
  loggers, managedthreads;


type TDownloadThread = class(TManagedThread)
 public
   constructor Create(url, targetPath, proxy, port : String; var logger : TLogger); overload;
   constructor Create(url, targetPath, targetFilename, proxy, port : String; var logger : TLogger); overload;
   function    getTargetFileName() : String;

 protected
    procedure Execute; override;

 private
    function getLogHeader : String;

    url_,
    targetPath_,
    targetFile_,
    proxy_,
    port_       : String;
    logger_     : TLogger;

    OutputFile_ : file;
end;


implementation

constructor TDownloadThread.Create(url, targetPath, proxy, port : String; var logger : TLogger); overload;
begin
  Create(url, targetPath, '', proxy, port, logger);
end;

constructor TDownloadThread.Create(url, targetPath, targetFilename, proxy, port : String; var logger : TLogger); overload;
var index : Longint;
begin
  inherited Create();

  url_ := url;
  targetPath_ := targetPath;
  targetFile_ := targetFileName;
  proxy_ := proxy;
  port_ := port;

  if Trim(targetFile_)='' then
  begin
    index := RPos('/', URI_);
    if index > 0 then
      targetFile_ := Copy(URI_, index+1, Length(URI_)-index);
    if Length(targetFile_) = 0 then
      targetFile_ := 'index.html';
    logger_.log(LVL_DEBUG, getLogHeader+'Target file set to '+targetFile_);
  end;

  logger_ := logger;
end;

function  TDownloadThread.getTargetFileName() : String;
begin
  Result := targetPath_+PathDelim+targetFile_;
end;

procedure TDownloadThread.execute();
var index       : Longint;
    AltFileName : String;
    Http        : THTTPSend;
begin
  logger_.log(LVL_DEBUG, getLogHeader+'Execute method started.');
  if FileExists(targetPath_+PathDelim+targetFile_) then
  begin
    index := 2;
    repeat
      AltFileName := targetFile_ + '.' + IntToStr(index);
      inc(index);
    until not FileExists(targetPath_+PathDelim+AltFileName);
    logger_.log(LVL_WARNING, getLogHeader+'"'+targetFile_+'" exists, writing to "'+AltFileName+'"');
    targetFile_ := AltFileName;
  end;

  HTTP := THTTPSend.Create;
  //HTTP.Document := targetPath_+PathDelim+targetFile_;
  if Trim(proxy_)<>'' then HTTP.ProxyHost := proxy_;
  if Trim(port_)<>'' then HTTP.ProxyPort := port_;
  //HTTP.Timeout := 20;
  HTTP.UserAgent := 'Mozilla/4.0 (compatible; Synapse for GPU at http://gpu.sourceforge.net)';
  logger_.log(LVL_DEBUG, getLogHeader+'User agent is '+HTTP.UserAgent);

  try
    if not HTTP.HTTPMethod('GET', Paramstr(1)) then
      begin
	logger_.log(LVL_SEVERE, getLogHeader+Http.Resultcode);
      end
    else
      begin
        logger_.log(LVL_DEBUG+getLogHeader+'HTTP Result was '+Http.Resultcode+' '+Http.Resultstring);
        logger_.log(LVL_DEBUG, Http.headers.text);
        logger_.log(LVL_INFO, 'New file created at '+Http.Document);
     end;
  finally
    HTTP.Free;
  end;

  logger_.log(LVL_DEBUG, getLogHeader+'Execute method finished.');
end;

function TDownloadThread.getLogHeader : String;
begin
 Result := 'DownloadThread ['+targetFile_+']> ';
end;


end.
