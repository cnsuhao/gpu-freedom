unit receiveserverservices;
{

  This unit receives a list of active servers from GPU II superserver
   and stores it in the TDbServerTable object.

  (c) 2010-2013 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, dbtablemanagers,
     servertables, loggers, downloadutils, coreconfigurations, geoutils,
     Classes, SysUtils, DOM, identities;

type TReceiveServerServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);

 protected
    procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveServerServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                   var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveServerServiceThread]> ', conf, tableman);
end;

procedure TReceiveServerServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbrow    : TDbServerRow;
    node     : TDOMNode;
    port     : String;

begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbrow.serverid  :=   node.FindNode('serverid').TextContent;
               dbrow.servername  := node.FindNode('servername').TextContent;
               dbrow.serverurl   := node.FindNode('serverurl').TextContent;
               dbrow.chatchannel := node.FindNode('chatchannel').TextContent;
               dbrow.version     := node.FindNode('version').TextContent;
               dbrow.online      := true;
               dbrow.updated     := true;
               dbrow.defaultsrv  := false;
               dbrow.superserver := node.FindNode('superserver').TextContent='true';
               dbrow.uptime      := StrToInt(node.FindNode('uptime').TextContent);
               dbrow.longitude   := StrToFloatDef(node.FindNode('longitude').TextContent, 0);
               dbrow.latitude    := StrToFloatDef(node.FindNode('latitude').TextContent, 0);
               dbrow.distance    := getDistanceOnEarthSphere(dbrow.longitude, dbrow.latitude,
                                                             myGPUId.Longitude, myGPUId.Latitude);
               dbrow.activenodes := StrToInt(node.FindNode('activenodes').TextContent);
               dbrow.jobinqueue := StrToInt(node.FindNode('jobinqueue').TextContent);
               dbrow.failures    := 0;

               tableman_.getServerTable().insertOrUpdate(dbrow);
               logger_.log(LVL_DEBUG, 'Updated or added <'+dbrow.servername+'> to tbserver table (distance: '+FloatToStr(dbrow.distance)+').');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, logHeader_+'Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;



procedure TReceiveServerServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/supercluster/list_servers.php?xml=1', xmldoc, false);
 if not erroneous_ then
    begin
     tableman_.getServerTable().execSQL('UPDATE tbserver set updated=0;');
     parseXml(xmldoc);
     if not erroneous_ then
       begin
        {
        tableman_.getServerTable().execSQL('UPDATE tbserver set online=updated;');
        tableman_.getServerTable().execSQL('UPDATE tbserver set defaultsrv=0;');
        tableman_.getServerTable().execSQL('UPDATE tbserver set defaultsrv=1 where distance=(select min(distance) from tbserver);');
        servMan_.reloadServers();
        }
       end;
    end;

 finishReceive('Service updated table TBSERVER succesfully :-)', xmldoc);
end;



end.
