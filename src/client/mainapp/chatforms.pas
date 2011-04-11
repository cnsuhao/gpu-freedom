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
    thread : TTransmitChannelServiceThread;
begin
  serverman.getDefaultServer(srv);
  thread := servicefactory.createTransmitChannelService(srv, 'Altos', 'CHAT', mmSubmitChat.Text);
  slot := serviceman.launch(thread);
  if (slot<>-1) then
     begin
       mmChat.Append(mmSubmitChat.Text);
       mmSubmitChat.Clear;
     end
      else
        begin
          // attempt to send chat failed
          thread.Free;
        end;
end;

initialization
  {$I chatforms.lrs}

end.

