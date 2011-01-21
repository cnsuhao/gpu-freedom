unit uploadutils;

interface

uses
  sysutils, strutils, httpsend, downloadutils, Classes, loggers;

const
  HTTP_UPLOAD_TIMEOUT = 60000; // 60 seconds

function uploadFromFile(url : AnsiString; sourcePath, sourceFile, proxy, port, logHeader : String; var logger : TLogger) : Boolean;
function uploadFromStream(url : AnsiString; proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;
function uploadFromFileOrStream(url : AnsiString; sourcePath, sourceFile, proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;

implementation

function uploadFromFile(url : AnsiString; sourcePath, sourceFile, proxy, port, logHeader : String; var logger : TLogger) : Boolean;
var dummy : TMemoryStream;
begin
 dummy := nil;
 Result := uploadFromFileOrStream(url, sourcePath, sourceFile, proxy, port, logHeader, logger, dummy);
end;

function uploadFromStream(url : AnsiString; proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;
begin
 Result := uploadFromFileOrStream(url, '', '', proxy, port, logHeader, logger, stream);
end;

function uploadFromFileOrStream(url : AnsiString; sourcePath, sourceFile, proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;
var
    Http        : THTTPSend;
    fromFile    : Boolean;
    temp        : TMemoryStream;
begin
  Result   := false;
  fromFile := (stream = nil);
  logger.log(LVL_DEBUG, logHeader+'Execute method started.');
  if fromFile then
    logger.log(LVL_INFO, logHeader+'Pushing file '+sourcePath+sourceFile+' to URL: '+url)
  else
    logger.log(LVL_INFO, logHeader+'Pushing memory stream to URL: '+url);

  HTTP := THTTPSend.Create;
  HTTP.Timeout   := HTTP_UPLOAD_TIMEOUT;
  HTTP.UserAgent := HTTP_USER_AGENT;

  if Trim(proxy)<>'' then HTTP.ProxyHost := proxy;
  if Trim(port)<>'' then HTTP.ProxyPort := port;

  logger.log(LVL_DEBUG, logHeader+'User agent is '+HTTP.UserAgent);

  if fromFile then
    HTTP.Document.LoadFromFile(sourcePath+sourceFile)
  else
    HTTP.Document.LoadFromStream(stream);

  try
    if not HTTP.HTTPMethod('POST', url) then
      begin
	logger.log(LVL_SEVERE, 'HTTP Error '+logHeader+IntToStr(Http.Resultcode)+' '+Http.Resultstring);
        Result := false;
      end
    else
        Result := parseHttpResult(HTTP, logger, logHeader);

  except
    on E : Exception do
      begin
       Result := false;
       logger.log(LVL_SEVERE, logHeader+'Exception '+E.Message+' thrown.');
      end;
  end;

  HTTP.Free;
end;


end.
