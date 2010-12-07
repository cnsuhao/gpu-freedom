unit receiveserverservices;
{

  This unit receives a list of active servers from GPU II superserver
   and stores it in the TDbServerTable object.

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers,
     servertables, loggers, downloadutils,
     XMLRead, DOM, Classes, SysUtils;

type TReceiveServerServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String;
                     servertable : TDbServerTable; var logger : TLogger);
 protected
    procedure Execute; override;

 private
   servertable_ : TDbServerTable;

   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveServerServiceThread.Create(var servMan : TServerManager; proxy, port : String;
                                             servertable : TDbServerTable; var logger : TLogger);
begin
 inherited Create(servMan, proxy, port, logger);
 servertable_ := servertable;
end;


procedure TReceiveServerServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbnode : TDbServerRow;
    node   : TDOMNode;
    port   : String;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;
  {
  while Assigned(node) do
    begin
        try
             begin
               dbnode.nodeid :=node.FindNode('nodeid').TextContent;
               //TODO: dbnode.defaultserver_id
               dbnode.nodename :=node.FindNode('processor').TextContent;
               dbnode.country :=node.FindNode('country').TextContent;
               dbnode.region :=node.FindNode('region').TextContent;
               dbnode.ip :=node.FindNode('ip').TextContent;
               port := node.FindNode('port').TextContent;
               if port='' then port:='0';
               dbnode.port :=StrToInt(port);
               //TODO: EAccessViolation dbnode.localip :=node.FindNode('localip').TextContent;
               dbnode.os :=node.FindNode('operatingsystem').TextContent;
               dbnode.cputype :=node.FindNode('cputype').TextContent;
               dbnode.version :=node.FindNode('version').TextContent;
               dbnode.acceptincoming :=(node.FindNode('accept').TextContent='1');
               dbnode.gigaflops :=StrToInt(node.FindNode('speed').TextContent);
               dbnode.ram :=StrToInt(node.FindNode('ram').TextContent);
               dbnode.mhz :=StrToInt(node.FindNode('mhz').TextContent);
               //dbnode.nbcpus :=StrToInt(node.FindNode('cpus').TextContent); empty
               dbnode.online := true;
               dbnode.updated := true;
               dbnode.uptime :=StrToFloatDef(node.FindNode('uptime').TextContent, 0);
               dbnode.totaluptime :=StrToFloatDef(node.FindNode('totuptime').TextContent, 0);
               dbnode.longitude :=StrToFloatDef(node.FindNode('geolocation_x').TextContent, 0);
               dbnode.latitude :=StrToFloatDef(node.FindNode('geolocation_y').TextContent, 0);
               nodetable_.insertOrUpdate(dbnode);
               logger_.log(LVL_DEBUG, 'Updated or added <'+dbnode.nodename+'> to tbnode table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)
   }
   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;



procedure TReceiveServerServiceThread.Execute;
var xmldoc    : TXMLDocument;
    stream    : TMemoryStream;
    proxyseed : String;
begin
 stream  := TMemoryStream.Create;

 proxyseed  := getProxySeed;
 erroneous_ := not downloadToStream(servMan_.getSuperServerUrl()+'/supercluster/get_servers.php?randomseed='+proxyseed,
               proxy_, port_, '[TReceiveServerServiceThread]> ', logger_, stream);

 if not erroneous_ then
 begin
  try
    stream.Position := 0; // to avoid Document root is missing exception
    xmldoc := TXMLDocument.Create();
    ReadXMLFile(xmldoc, stream);
  except
     on E : Exception do
        begin
           erroneous_ := true;
           logger_.log(LVL_SEVERE, '[TReceiveServerServiceThread]> Exception catched in Execute: '+E.Message);
        end;
  end; // except

  if not erroneous_ then
    begin
     servertable_.execSQL('UPDATE tbserver set updated=0;');
     parseXml(xmldoc);
     if not erroneous_ then
       servertable_.execSQL('UPDATE tbserver set online=updated;');
    end;
  xmldoc.Free;
 end;


 if stream <>nil then stream.Free  else logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Internal error in receiveserverservices.pas, stream is nil');

 done_ := true;
end;



end.
