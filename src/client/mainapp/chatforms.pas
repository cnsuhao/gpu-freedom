unit chatforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, StdCtrls, servicefactories, coreobjects, transmitchannelservices,
  servermanagers;

type

  { TChatForm }

  TChatForm = class(TForm)
    btnSend: TButton;
    cbSelectChannel: TComboBox;
    mmChat: TMemo;
    mmSubmitChat: TMemo;
    PanelBottom: TPanel;
    PanelTop: TPanel;
    procedure btnSendClick(Sender: TObject);
  private
    thread : TTransmitChannelServiceThread;
  public
    { public declarations }
  end; 

var
  ChatForm: TChatForm;

implementation

{ TChatForm }

procedure TChatForm.btnSendClick(Sender: TObject);
var srv  : TServerRecord;
    slot : Longint;
begin
  serverman.getDefaultServer(srv);
  ShowMessage(srv.url);
  if (thread<>nil) and (thread.isDone()) then thread.Free;
  if thread = nil then
     begin
       thread := servicefactory.createTransmitChannelService(srv, 'Altos', 'CHAT', mmSubmitChat.Text);
       slot := serviceman.launch(thread);
       if (slot<>-1) then mmSubmitChat.Clear;
     end;
end;

initialization
  {$I chatforms.lrs}

end.

