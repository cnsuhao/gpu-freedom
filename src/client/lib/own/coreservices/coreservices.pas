unit coreservices;
{

  This unit sets up the ancestor for each request and response done to the GPU II servers

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses
  managedthreads, servermanagers, loggers, downloadutils,
  dbtablemanagers, coreconfigurations,
  XMLRead, DOM, Classes, SysUtils;

type TCoreServiceThread = class(TManagedThread)
  public
    constructor Create(var logger : TLogger; logHeader : String; var conf : TCoreConfiguration; var tableman : TDbTableManager);

  protected
    logger_    : TLogger;
    logHeader_ : String;
    conf_     : TCoreConfiguration;
    tableman_ : TDbTableManager;
end;

type TCommServiceThread = class(TCoreServiceThread)
  public
    constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger; logHeader : String;
                       var conf : TCoreConfiguration; var tableman : TDbTableManager);

  protected
    srv_      : TServerRecord;
    servMan_  : TServerManager;
    proxy_,
    port_     : String;

    procedure finishComm(logSuccess : String);

end;


type TInternalServiceThread = class(TCoreServiceThread)
end;

type TTransmitServiceThread = class(TCommServiceThread)
   procedure transmit(request : AnsiString; noargs : Boolean);
   procedure finishTransmit(logSuccess : String);
end;


type TReceiveServiceThread = class(TCommServiceThread)
   procedure receive(request : AnsiString; var xmldoc : TXmlDocument; noargs : Boolean);
   procedure finishReceive(logSuccess : String; var xmldoc : TXmlDocument);
end;



implementation

constructor TCoreServiceThread.Create(var logger : TLogger; logHeader : String;
                                      var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
  inherited Create(true); // suspended
  logger_  := logger;
  logHeader_ := logHeader;
  conf_     := conf;
  tableman_ := tableman;
end;

constructor TCommServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger; logHeader : String;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
  inherited Create(logger, logHeader, conf, tableman);
  srv_      := srv;
  servMan_  := servMan;
  proxy_    := proxy;
  port_     := port;
end;


procedure TTransmitServiceThread.transmit(request : AnsiString; noargs : Boolean);
var
    stream    : TMemoryStream;
begin
 stream  := TMemoryStream.Create;
 erroneous_ := not downloadToStream(srv_.url+request+getProxyArg(noargs),
               proxy_, port_, logHeader_, logger_, stream);

 if stream <>nil then stream.Free  else logger_.log(LVL_SEVERE,
         logHeader_+'Internal error in coreservices.pas, stream is nil');
end;


procedure TReceiveServiceThread.receive(request : AnsiString; var xmldoc : TXmlDocument; noargs : Boolean);
var
    stream    : TMemoryStream;
begin
 xmldoc := TXMLDocument.Create();
 stream  := TMemoryStream.Create;
 erroneous_ := not downloadToStream(srv_.url+request+getProxyArg(noargs),
               proxy_, port_, logHeader_, logger_, stream);

 if stream=nil then
  begin
   logger_.log(LVL_SEVERE, logHeader_+'Internal error in coreservices.pas, stream is nil');
   erroneous_ := true;
  end;

 if not erroneous_ then
 begin
  try
    stream.Position := 0; // to avoid Document root is missing exception
    ReadXMLFile(xmldoc, stream);
  except
     on E : Exception do
        begin
           erroneous_ := true;
           logger_.log(LVL_SEVERE, logHeader_+'Exception catched in Execute: '+E.Message);
        end;
  end; // except
  end; // if not erroneous
  if stream<>nil then stream.Free;
end;

procedure TCommServiceThread.finishComm(logSuccess : String);
begin
 if erroneous_ then
    begin
     servMan_.increaseFailures(srv_.url);
     logger_.log(LVL_SEVERE, logHeader_+'Service finished but ERRONEOUS flag set :-(')
    end
 else
   logger_.log(LVL_INFO, logHeader_+logSuccess);
 done_ := true;
end;


procedure TReceiveServiceThread.finishReceive(logSuccess : String; var xmldoc : TXmlDocument);
begin
 if xmldoc=nil then
   begin
    logger_.log(LVL_SEVERE, logHeader_+'xmldoc is nil in finalize');
    erroneous_ := true;
   end
 else
    xmldoc.Free;

 finishComm(logSuccess);
end;

procedure TTransmitServiceThread.finishTransmit(logSuccess : String);
begin
 finishComm(logSuccess);
end;


end.
