unit receivechannelservices;
{

  This unit receives the content of a channel from a GPU II server

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers, coreconfigurations,
     channeltables, retrievedtables, dbtablemanagers,
     loggers, identities, Classes, SysUtils, DOM, synacode;

type TReceiveChannelServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager;
                     channame, chantype : String);
 protected
    procedure Execute; override;

 private
   channame_,
   chantype_ : String;

   function  getPHPArguments(var row : TDbRetrievedRow) : AnsiString;
   procedure parseXml(var xmldoc : TXMLDocument; var row : TDbRetrievedRow);
end;

implementation

constructor TReceiveChannelServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager;
                     channame, chantype : String);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveChannelServiceThread]> ', conf, tableman);
 channame_ := channame;
 chantype_ := chantype;
end;

function  TReceiveChannelServiceThread.getPHPArguments(var row : TDbRetrievedRow) : AnsiString;
begin
 Result :=  'nodeid='+encodeUrl(myGPUId.NodeId)+'&lastmsg='+IntToStr(row.lastmsg)+'&chantype='+encodeURl(chantype_)+'&channame='+encodeUrl(channame_);
end;

procedure TReceiveChannelServiceThread.parseXml(var xmldoc : TXMLDocument; var row : TDbRetrievedRow);
var
    dbnode   : TDbChannelRow;
    node     : TDOMNode;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbnode.content           := node.FindNode('content').TextContent;
               dbnode.server_id         := srv_.id;
               dbnode.externalid        := StrToInt(node.FindNode('id').TextContent);
               dbnode.nodename          := node.FindNode('nodename').TextContent;
               dbnode.nodeid            := node.FindNode('nodeid').TextContent;
               dbnode.user              := node.FindNode('user').TextContent;
               dbnode.channame          := channame_;
               dbnode.chantype          := chantype_;
               dbnode.create_dt         := Now();
               dbnode.usertime_dt       := Now(); //TODO parse from string from server

               if dbnode.externalid>row.lastmsg then row.lastmsg := dbnode.externalid;
               logger_.log(LVL_DEBUG, logHeader_+'Adding message '+IntToStr(dbnode.externalid)+' to tbchannel table.');
               tableman_.getChannelTable().insert(dbnode);
               logger_.log(LVL_DEBUG, 'record count: '+IntToStr(tableman_.getChannelTable().getDS().RecordCount));
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

   tableMan_.getRetrievedTable.insertOrUpdate(row);
   logger_.log(LVL_DEBUG, logHeader_+'Parameter in TBRETRIEVED updated with lastmsg '+IntToStr(row.lastmsg)+', msgtype '+row.msgtype+'.');
   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;


procedure TReceiveChannelServiceThread.Execute;
var xmldoc    : TXMLDocument;
    row       : TDbRetrievedRow;
begin
 tableman_.getRetrievedTable().getRow(row, srv_.id, channame_, chantype_);

 receive('/channel/get_channel_messages_xml.php?'+getPHPArguments(row), xmldoc, false);

 if not erroneous_ then
     parseXml(xmldoc, row);

 finishReceive('Service updated table TBCHANNEL succesfully :-)', xmldoc);
end;


end.
