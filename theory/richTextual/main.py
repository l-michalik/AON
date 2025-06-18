import psutil
from textual.app import App, ComposeResult # type: ignore
from textual.widgets import Header, Footer, Static # type: ignore
from textual.containers import Vertical # type: ignore
from textual.reactive import reactive # type: ignore

class CPUMemoryWidget(Static):
    cpu: reactive[float] = reactive(0.0)
    memory: reactive[float] = reactive(0.0)
    
    async def on_mount(self) -> None:
        self.set_interval(1, self.update_stats)
        
    def update_stats(self) -> None:
        self.cpu = psutil.cpu_percent(interval=1)
        self.memory = psutil.virtual_memory().percent
        
    def render(self) -> str:
        return (
            f"[bold green]CPU Usage:[/bold green] {self.cpu}%\n"
            f"[bold blue]Memory Usage:[/bold blue] {self.memory}%"
        )
        
class RichTextualApp(App):
    TITLE = "Rich Textual App"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh Stats"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(CPUMemoryWidget(), id="stats")
        yield Footer()
    
    
if __name__ == "__main__":
    app = RichTextualApp()
    app.run()