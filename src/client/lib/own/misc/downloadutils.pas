unit downloadutils;


interface

uses
  sysutils, strutils, httpsend, Classes, loggers;

const
  HTTP_DOWNLOAD_TIMEOUT = 10000; // 10 seconds
  HTTP_USER_AGENT = 'Mozilla/4.0 (compatible; Synapse for GPU at http://gpu.sourceforge.net)';


function downloadToFile(url, targetPath, targetFile, proxy, port, logHeader : String; var logger : TLogger) : Boolean;
function downloadToStream(url, proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;
function downloadToFileOrStream(url, targetPath, targetFile, proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;

implementation

function downloadToFile(url, targetPath, targetFile, proxy, port, logHeader : String; var logger : TLogger) : Boolean;
var dummy : TMemoryStream;
begin
 dummy := nil;
 Result := downloadToFileOrStream(url, targetPath, targetFile, proxy, port, logHeader, logger, dummy);
end;

function downloadToStream(url, proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;
begin
 Result := downloadToFileOrStream(url, '', '', proxy, port, logHeader, logger, stream);
end;


function downloadToFileOrStream(url, targetPath, targetFile, proxy, port, logHeader : String; var logger : TLogger; var stream : TMemoryStream) : Boolean;
var
    Http        : THTTPSend;
    saveFile    : Boolean;
begin
  Result   := true;
  saveFile := (stream = nil);
  logger.log(LVL_DEBUG, logHeader+'Execute method started.');
  logger.log(LVL_DEBUG, logHeader+'Retrieving data from URL: '+url);

  HTTP := THTTPSend.Create;
  HTTP.Timeout   := HTTP_DOWNLOAD_TIMEOUT;
  HTTP.UserAgent := HTTP_USER_AGENT;

  if Trim(proxy)<>'' then HTTP.ProxyHost := proxy;
  if Trim(port)<>'' then HTTP.ProxyPort := port;

  logger.log(LVL_DEBUG, logHeader+'User agent is '+HTTP.UserAgent);

  try
    if not HTTP.HTTPMethod('GET', url) then
      begin
	logger.log(LVL_SEVERE, 'HTTP Error '+logHeader+IntToStr(Http.Resultcode)+' '+Http.Resultstring);
        Result := false;
      end
    else
      begin
        logger.log(LVL_DEBUG, logHeader+'HTTP Result was '+IntToStr(Http.Resultcode)+' '+Http.Resultstring);
        logger.log(LVL_DEBUG, logHeader+'HTTP Header is ');
        logger.log(LVL_DEBUG, Http.headers.text);
        if saveFile then
            begin
               HTTP.Document.SaveToFile(targetPath+targetFile);
               logger.log(LVL_INFO, logHeader+'New file created at '+targetPath+targetFile);
            end
           else
            begin
               HTTP.Document.SaveToStream(stream);
               logger.log(LVL_INFO, logHeader+'New stream created');
            end;
      end;

  except
    on E : Exception do
      begin
       Result := false;
       logger.log(LVL_SEVERE, logHeader+'Exception '+E.Message+' thrown.');
      end;
  end;

  HTTP.Free;
  logger.log(LVL_DEBUG, logHeader+'Execute method finished.');
end;


end.
