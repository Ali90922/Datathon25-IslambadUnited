import { Link } from "react-router-dom";

const Header = () => {
	const navLinks = [
		{ name: "Home", path: "/" },
		{ name: "Visualize", path: "/visualize" },
	];

	return (
		<nav className='bg-primary'>
			<ul className='flex items-center justify-center'>
				{navLinks.map((link, index) => (
					<Link key={index} to={link.path} className='p-4 hover:bg-secondary'>
						{link.name}
					</Link>
				))}
			</ul>
		</nav>
	);
};

export default Header;
