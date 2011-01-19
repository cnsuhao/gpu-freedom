unit servicefactories;
 {
   TServiceFactory creates all services available to the GPU core.servicemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  servermanagers, dbtablemanagers, loggers,
  receiveclientservices, transmitclientservices,
  receiveserverservices, receiveparamservices,
  receivechannelservices, transmitchannelservices,
  receivejobservices,
  coreconfigurations;

type TServiceFactory = class(TObject)
   public
    constructor Create(var servMan : TServerManager;
                       var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration);
    destructor Destroy;

    function createReceiveClientService(var srv : TServerRecord)  : TReceiveClientServiceThread;
    function createTransmitClientService(var srv : TServerRecord) : TTransmitClientServiceThread;
    function createReceiveServerService(var srv : TServerRecord)  : TReceiveServerServiceThread;
    function createReceiveParamService(var srv : TServerRecord)   : TReceiveParamServiceThread;
    function createReceiveChannelService(var srv : TServerRecord;
                                         channame, chantype : String) : TReceiveChannelServiceThread;
    function createTransmitChannelService(var srv : TServerRecord;
                                          channame, chantype : String;
                                          content : AnsiString) : TTransmitChannelServiceThread;
    function createReceiveJobService(var srv : TServerRecord) : TReceiveJobServiceThread;

   private

     servMan_  : TServerManager;
     tableMan_ : TDbTableManager;
     logger_   : TLogger;
     proxy_,
     port_     : String;
     conf_     : TCoreConfiguration;

end;

implementation

constructor TServiceFactory.Create(var servMan : TServerManager;
                                   var tableMan : TDbTableManager; proxy, port : String; var logger : TLogger; var conf : TCoreConfiguration);
begin
 servMan_  := servMan;
 tableMan_ := tableMan;
 logger_   := logger;

 proxy_    := proxy;
 port_     := port;
 conf_     := conf;
end;

destructor TServiceFactory.Destroy;
begin
end;

function TServiceFactory.createReceiveClientService(var srv : TServerRecord) : TReceiveClientServiceThread;
begin
 Result := TReceiveClientServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createTransmitClientService(var srv : TServerRecord) : TTransmitClientServiceThread;
begin
 Result := TTransmitClientServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveServerService(var srv : TServerRecord)  : TReceiveServerServiceThread;
begin
  Result := TReceiveServerServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveParamService(var srv : TServerRecord) : TReceiveParamServiceThread;
begin
 Result := TReceiveParamServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;

function TServiceFactory.createReceiveChannelService(var srv : TServerRecord;
                                     channame, chantype : String) : TReceiveChannelServiceThread;
begin
  Result := TReceiveChannelServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, channame, chantype);
end;

function TServiceFactory.createTransmitChannelService(var srv : TServerRecord;
                                                      channame, chantype : String;
                                                      content : AnsiString) : TTransmitChannelServiceThread;
begin
 Result := TTransmitChannelServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_, channame, chantype, content);
end;

function TServiceFactory.createReceiveJobService(var srv : TServerRecord) : TReceiveJobServiceThread;
begin
 Result := TReceiveJobServiceThread.Create(servMan_, srv, proxy_, port_, logger_, conf_, tableman_);
end;


end.
