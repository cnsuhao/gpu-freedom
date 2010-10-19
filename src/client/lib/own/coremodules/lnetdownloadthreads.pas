unit lnetdownloadthreads;
{
  DownloadThread is a thread which downloads a file
  via HTTP and stores it in targetfile under directory
  targetDir.

  This file is build with fpget as template, which belongs to the
  lnet project under http://lnet.wordpress.com/
  It uses LNet to download the file via HTTP.

  (c) by 2002-2010 the GPU Development Team
  This unit is released under GNU Public License (GPL)
}

interface

uses
  sysutils, strutils, lnet, lhttp, lHTTPUtil, lnetSSL,
  loggers, managedthreads;


type TLNETDownloadThread = class(TManagedThread)
 public
   constructor Create(url, targetPath : String; var logger : TLogger); overload;
   constructor Create(url, targetPath, targetFilename : String; var logger : TLogger); overload;
   function    getTargetFileName() : String;

   // handlers for TLHTTPClient events
   procedure ClientDisconnect(ASocket: TLSocket);
   procedure ClientDoneInput(ASocket: TLHTTPClientSocket);
   procedure ClientError(const Msg: AnsiString; aSocket: TLSocket);
   function ClientInput(ASocket: TLHTTPClientSocket; ABuffer: pchar;
      ASize: Integer): Integer;
   procedure ClientProcessHeaders(ASocket: TLHTTPClientSocket);

 protected
    procedure Execute; override;

 private
    function getLogHeader : String;

    url_,
    targetPath_,
    targetFile_ : String;
    logger_     : TLogger;

    host_,
    uri_        : AnsiString;
    port_       : Word;
    useSSL_     : Boolean;

    HttpClient_ : TLHTTPClient;
    SSLSession_ : TLSSLSession;
    OutputFile_ : file;
end;


implementation

constructor TLNETDownloadThread.Create(url, targetPath : String; var logger : TLogger); overload;
begin
  Create(url, targetPath, '', logger);
end;

constructor TLNETDownloadThread.Create(url, targetPath, targetFilename : String; var logger : TLogger); overload;
var index : Longint;
begin
  inherited Create();

  url_ := url;
  targetPath_ := targetPath;
  targetFile_ := targetFileName;

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

function  TLNETDownloadThread.getTargetFileName() : String;
begin
  Result := targetPath_+PathDelim+targetFile_;
end;

procedure TLNETDownloadThread.execute();
var index : Longint;
    AltFileName : String;
begin
  UseSSL_ := DecomposeURL(URL_, Host_, URI_, Port_);
  logger_.log(LVL_INFO, getLogHeader+'Host: '+Host_+' URI: '+URI_+' Port: '+IntToStr(Port_));

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

  try
    assign(OutputFile_, targetPath_+PathDelim+targetFile_);
    rewrite(OutputFile_, 1);
  except
    CloseFile(OutputFile_);
    done_ := true;
    erroneous_ := true;
    logger_.log(LVL_SEVERE, getLogHeader+'Serious problem in opening target file '+targetPath_+PathDelim+targetFile_);
    Exit;
  end;

  HttpClient_ := TLHTTPClient.Create(nil);

  SSLSession_ := TLSSLSession.Create(HttpClient_);
  SSLSession_.SSLActive := UseSSL_;

  HttpClient_.Session := SSLSession_;
  HttpClient_.Host := Host_;
  HttpClient_.Method := hmGet;
  HttpClient_.Port := Port_;
  HttpClient_.URI := URI_;
  HttpClient_.Timeout := 1;
  HttpClient_.OnDisconnect := @self.ClientDisconnect;
  HttpClient_.OnDoneInput := @self.ClientDoneInput;
  HttpClient_.OnError := @self.ClientError;
  HttpClient_.OnInput := @self.ClientInput;
  HttpClient_.OnProcessHeaders := @self.ClientProcessHeaders;
  HttpClient_.SendRequest;
  done_ := false;

  try
   while not done_ do
    HttpClient_.CallAction;
  finally
    HttpClient_.Free;
  end;
  logger_.log(LVL_INFO, getLogHeader+'Execute method finished');
end;

function TLNETDownloadThread.getLogHeader : String;
begin
 Result := 'DownloadThread ['+targetFile_+']> ';
end;

procedure TLNETDownloadThread.ClientError(const Msg: AnsiString; aSocket: TLSocket);
begin
  erroneous_ := true;
  logger_.log(LVL_WARNING, getLogHeader+'Error: '+Msg);
end;

procedure TLNETDownloadThread.ClientDisconnect(ASocket: TLSocket);
begin
  logger_.log(LVL_DEBUG, getLogHeader+'Disconnected.');
  done_ := true;
end;

procedure TLNETDownloadThread.ClientDoneInput(ASocket: TLHTTPClientSocket);
begin
  logger_.log(LVL_DEBUG, getLogHeader+'Closing outputfile and disconnecting socket...');
  close(OutputFile_);
  ASocket.Disconnect;
  logger_.log(LVL_DEBUG, getLogHeader+'Closing outputfile and disconnecting socket done.');
end;

function TLNETDownloadThread.ClientInput(ASocket: TLHTTPClientSocket;
  ABuffer: pchar; ASize: Integer): Integer;
begin
  blockwrite(outputfile_, ABuffer^, ASize, Result);
  logger_.log(LVL_DEBUG, getLogHeader+IntToStr(ASize) + 'bytes received...');
end;

procedure TLNETDownloadThread.ClientProcessHeaders(ASocket: TLHTTPClientSocket);
begin
  logger_.log(LVL_DEBUG, getLogHeader+'Response: '+IntToStr(HTTPStatusCodes[ASocket.ResponseStatus])+' '+
    ASocket.ResponseReason+', data...');
end;


end.
