unit receivechannelservices;
{

  This unit receives the content of a channel from a GPU II server

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers,
     channeltables, retrievedtables, dbtablemanagers,
     loggers, Classes, SysUtils, DOM;

type TReceiveChannelServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan  : TServerManager; proxy, port : String;
                     var tableman : TDbTableManager; var logger : TLogger;
                     var srv : TServerRecord; channame, chantype : String);
 protected
    procedure Execute; override;

 private
   tableman_ : TDbTableManager;
   srv_      : TServerRecord;
   channame_,
   chantype_ : String;

   function  getPHPArguments(var row : TDbRetrievedRow) : AnsiString;
   procedure parseXml(var xmldoc : TXMLDocument; var srv : TServerRecord; var row : TDbRetrievedRow);
end;

implementation

constructor TReceiveChannelServiceThread.Create(var servMan  : TServerManager; proxy, port : String;
                                                var tableman : TDbTableManager; var logger : TLogger;
                                                var srv : TServerRecord; channame, chantype : String);
begin
  inherited Create(servMan, proxy, port, logger);
  tableman_ := tableman;
  srv_      := srv;
  channame_ := channame;
  chantype_ := chantype;
end;

function  TReceiveChannelServiceThread.getPHPArguments(var row : TDbRetrievedRow) : AnsiString;
begin
 Result := 'chantype='+chantype_+'&channame='+channame_+'&lastmsg='+IntToStr(row.lastmsg);
end;

procedure TReceiveChannelServiceThread.parseXml(var xmldoc : TXMLDocument; var srv : TServerRecord;
                                                var row : TDbRetrievedRow);
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
               dbnode.server_id         := srv.id;
               dbnode.externalid        := StrToInt(node.FindNode('externalid').TextContent);
               dbnode.nodename          := node.FindNode('nodename').TextContent;
               dbnode.nodeid            := node.FindNode('nodeid').TextContent;
               dbnode.user              := node.FindNode('user').TextContent;
               dbnode.channame          := channame_;
               dbnode.chantype          := chantype_;
               dbnode.create_dt         := Now();
               dbnode.usertime_dt       := Now(); //TODO parse from string from server

               tableman_.getChannelTable().insert(dbnode);
               if dbnode.externalid>row.lastmsg then row.lastmsg := dbnode.externalid;
               logger_.log(LVL_DEBUG, 'Updated or added message '+IntToStr(dbnode.id)+' to tbchannel table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, '[TReceiveChannelServiceThread]> Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)

   tableMan_.getRetrievedTable.insertOrUpdate(row);
   logger_.log(LVL_DEBUG, 'Parameter in TBRETRIEVED updated with lastmsg '+IntToStr(row.lastmsg)+', msgtype '+row.msgtype+'.');
   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;


procedure TReceiveChannelServiceThread.Execute;
var xmldoc    : TXMLDocument;
    row       : TDbRetrievedRow;
begin
 tableman_.getRetrievedTable().getRow(row, srv_.id, channame_, chantype_);

 receive(srv_, '/channel/get_channel_xml.php?'+getPHPArguments(row),
         '[TReceiveChannelServiceThread]> ', xmldoc, true);

 if not erroneous_ then
     parseXml(xmldoc, srv_, row);

 finishReceive(srv_, '[TReceiveChannelServiceThread]> ', 'Service updated table TBCHANNEL succesfully :-)', xmldoc);
end;


end.
