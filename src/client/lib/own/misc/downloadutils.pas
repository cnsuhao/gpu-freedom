unit downloadutils;


interface

uses
  sysutils, strutils, httpsend, Classes, loggers;

const
  HTTP_DOWNLOAD_TIMEOUT = 10000; // 10 seconds
  HTTP_USER_AGENT = 'Mozilla/4.0 (compatible; Synapse for GPU at http://gpu.sourceforge.net)';


function downloadToFile(url, targetPath, targetFile, proxy, port, logHeader : String; var logger : TLogger) : Boolean;


implementation


function downloadToFile(url, targetPath, targetFile, proxy, port, logHeader : String; var logger : TLogger) : Boolean;
var index       : Longint;
    AltFileName : String;
    Http        : THTTPSend;
begin
  Result := true;
  logger.log(LVL_DEBUG, logHeader+'Execute method started.');
  logger.log(LVL_DEBUG, logHeader+'Retrieving data from URL: '+url);

  if FileExists(targetPath+targetFile) then
  begin
    index := 2;
    repeat
      AltFileName := targetFile + '.' + IntToStr(index);
      inc(index);
    until not FileExists(targetPath+AltFileName);
    logger.log(LVL_WARNING, logHeader+'"'+targetFile+'" exists, writing to "'+AltFileName+'"');
    targetFile := AltFileName;
  end;

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
        HTTP.Document.SaveToFile(targetPath+targetFile);
        logger.log(LVL_INFO, 'New file created at '+targetPath+targetFile);
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
