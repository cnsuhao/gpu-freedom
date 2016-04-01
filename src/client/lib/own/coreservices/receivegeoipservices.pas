unit receivegeoipservices;
{

  This unit receives GEO IP information from a GPU II server, stores them into local
  tbgeoip table and into myConfId structure.

  (c) 2010-2016 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}

interface

uses coreservices, servermanagers, dbtablemanagers,
     loggers, coreconfigurations, geoiptables,
     Classes, SysUtils, DOM, identities;

type TReceiveGeoIPServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);

 protected
   procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);
   procedure loadGeoIPIntoConfiguration;
end;



implementation

constructor TReceiveGeoIPServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                                              var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, proxy, port, logger, '[TReceiveGeoIPServiceThread]> ', conf, tableman);
end;

procedure TReceiveGeoIPServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    node     : TDOMNode;
    dbnode   : TDbGeoIPRow;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
 try
  begin
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
               if node.findNode('error')<>nil then
                      begin
                           dbnode.error := node.FindNode('error').TextContent;
                           dbnode.ip := node.FindNode('ip').TextContent;
                           dbnode.city:= '';
                           dbnode.countrycode:='';
                           dbnode.countryname:='';
                           dbnode.timezone:='';
                           dbnode.latitude:=0;
                           dbnode.longitude:=0;
                           dbnode.create_dt:= Now();
                      end
                else
                      begin
                           dbnode.error := '';
                           dbnode.ip := node.FindNode('ip').TextContent;
                           dbnode.city:= node.FindNode('city').TextContent;
                           dbnode.countrycode:=node.FindNode('countrycode').TextContent;
                           dbnode.countryname:=node.FindNode('countryname').TextContent;
                           dbnode.timezone:=node.FindNode('timezone').TextContent;
                           // TODO: handle formatting
                           dbnode.latitude:=StrToFloat(node.FindNode('latitude').TextContent);
                           dbnode.longitude:=StrToFloat(node.FindNode('longitude').TextContent);
                           dbnode.create_dt:= Now();
                      end;




               logger_.log(LVL_DEBUG, logHeader_+'Adding information '+dbnode.ip+' to tbgeoip table.');
               tableman_.getGeoIPTable().insert(dbnode);
               logger_.log(LVL_DEBUG, 'record count: '+IntToStr(tableman_.getGeoIPTable().getDS().RecordCount));

        node := node.NextSibling;
     end;  // while Assigned(param)

   end; // try
       except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, logHeader_+'Exception catched in parseXML: '+E.Message);
              end;
          end; // except


   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
   loadGeoIPIntoConfiguration;
   logger_.log(LVL_DEBUG, 'All parameters updated succesfully');
end;


procedure TReceiveGeoIPServiceThread.loadGeoIPIntoConfiguration;
begin
    with myGPUId do
                begin
                  longitude := tableman_.getGeoIPTable().getLongitude();
                  latitude := tableman_.getGeoIPTable().getLatitude();
                end; // with

end;

procedure TReceiveGeoIPServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/geoip/geoip.php?xml=1', xmldoc, false);

 if not erroneous_ then
     parseXml(xmldoc);

 finishReceive('Service retrieved GEO IP parameters from GPU server succesfully :-)', xmldoc);
end;




end.
