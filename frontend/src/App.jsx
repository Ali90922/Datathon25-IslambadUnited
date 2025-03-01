import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainLayout from "./components/MainLayout";
import Home from "./pages/Home";
import Visualize from "./pages/Visualize";

function App() {
	return (
		<Router>
			<MainLayout>
				<Routes>
					<Route path='/' element={<Home />} />
					<Route path='/visualize' element={<Visualize />} />
				</Routes>
			</MainLayout>
		</Router>
	);
}

export default App;
